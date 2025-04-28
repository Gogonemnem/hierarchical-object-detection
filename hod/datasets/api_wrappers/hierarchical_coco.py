from collections import defaultdict
import copy
import datetime
import time
from typing import Dict

import numpy as np
from pycocotools import mask as maskUtils

from mmdet.datasets.api_wrappers import COCOeval

from hod.utils.tree import HierarchyTree


class HierarchicalCOCOeval(COCOeval):
    """This class is a wrapper for COCOeval to support hierarchical evaluation.

    It adds the ability to compute hierarchical metrics based on a taxonomy tree.
    """

    def __init__(self, hierarchy_tree, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hierarchy_tree = HierarchyTree(hierarchy_tree)
        self.label_to_name = {cat['id']: cat['name'] for cat in self.cocoGt.dataset['categories']}
        self.params.useCats = 0
    
    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
        dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results
    
    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId): computeIoU(imgId) \
                        for imgId in p.imgIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, areaRng, maxDet)
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))
    
    def computeOks(self, imgId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId]
        dts = self._dts[imgId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def computeIoU(self, imgId):
        p = self.params
        gt = self._gts[imgId]
        dt = self._dts[imgId]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious


    def evaluateImg(self, imgId, aRng, maxDet):
        '''
        perform evaluation for single image
        :return: dict (single image results)
        '''
        p = self.params

        gt = self._gts[imgId]
        dt = self._dts[imgId]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId][:, gtind] if len(self.ious[imgId]) > 0 else self.ious[imgId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        hierarchical_stats = {
            'overlaps': np.zeros((T, D)),
            'len_dt': np.zeros((D)),
            'len_gt': np.zeros((G)),
        }
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                # flip loop order as multiple dts can match a single gt with hierarchical reasoning
                for gind, g in enumerate(gt):
                    # skip crowds just like COCOeval
                    if iscrowd[gind]:
                        continue
                    # information about best match so far (m=-1 -> unmatched)
                    hf1 = 0
                    iou = min([t,1-1e-10])
                    m   = -1 # best detection match index
                    len_overlap = 0
                    for dind, d in enumerate(dt):
                        # if this dt already matched, and not a crowd, continue
                        if dtm[tind, dind]>0:
                            continue

                        metrics = hierarchical_prf_metric(
                            self.hierarchy_tree, d['category_id'], g['category_id'],
                            self.label_to_name, return_paths=True
                        )
                        hierarchical_stats['len_dt'][dind] = metrics['len_dt']
                        hierarchical_stats['len_gt'][gind] = metrics['len_gt']

                        # looping over gt first, only leftover dts, no no need to check
                        # # if dt matched to reg gt, and on ignore gt, stop
                        # if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                        #     break

                        # continue to next gt unless better class match made
                        if metrics['hf1'] < hf1:
                            continue
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        hf1=metrics['hf1']
                        iou=ious[dind,gind]
                        m=dind
                        len_overlap = metrics['len_overlap']

                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,m]   = gtIg[gind]
                    dtm[tind,m]    = g['id']
                    gtm[tind,gind] = dt[m]['id']
                    hierarchical_stats['overlaps'][tind, m] = len_overlap
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image
        return {
                'image_id':     imgId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
                'hstats': hierarchical_stats,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        # K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,A,M))
        scores      = -np.ones((T,R,A,M))
        f1          = -np.ones((T,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        # setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        # k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        # A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        # for k, k0 in enumerate(k_list):
        #     Nk = k0*A0*I0
        for a, a0 in enumerate(a_list):
            Na = a0*I0
            for m, maxDet in enumerate(m_list):
                E = [self.evalImgs[Na + i] for i in i_list]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue
                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.
                inds = np.argsort(-dtScores, kind='mergesort')
                dtScoresSorted = dtScores[inds]

                # dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                
                # Hierarchical Stats
                overlaps = np.concatenate([e['hstats']['overlaps'][:, :maxDet] for e in E], axis=1)[:, inds]
                len_dt   = np.concatenate([e['hstats']['len_dt'][:maxDet]   for e in E], axis=0)[inds]
                len_gt   = np.concatenate([e['hstats']['len_gt'] for e in E], axis=0)

                # Mask ignored GTs
                valid_gt_mask = (gtIg == 0)  # shape: (G,)
                gt_lens_masked = len_gt[valid_gt_mask]  # keep only valid GTs

                # Now sum all nodes in valid GT paths (for each IoU threshold)
                npig = gt_lens_masked.sum(axis=0)
                if npig == 0:
                    continue
                
                valid_mask = ~dtIg
                tps = np.where(valid_mask, overlaps, 0)
                len_dt_2d = np.broadcast_to(len_dt, overlaps.shape)  # shape: (T, D)
                fps = np.where(valid_mask, len_dt_2d - overlaps, 0)

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    nd = len(tp)
                    rc = tp / npig
                    pr = tp / (fp+tp+np.spacing(1))
                    q  = np.zeros((R,))
                    ss = np.zeros((R,))

                    if nd:
                        recall[t,a,m] = rc[-1]
                        
                    else:
                        recall[t,a,m] = 0
                    f1_curve = 2 * pr * rc / (pr + rc + 1e-6)
                    f1[t, a, m] = np.max(f1_curve)

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    pr = pr.tolist(); q = q.tolist()

                    for i in range(nd-1, 0, -1):
                        if pr[i] > pr[i-1]:
                            pr[i-1] = pr[i]

                    inds = np.searchsorted(rc, p.recThrs, side='left')
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                            ss[ri] = dtScoresSorted[pi]
                    except:
                        pass
                    precision[t,:,a,m] = np.array(q)
                    scores[t,:,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
            'f1': f1,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( metric='precision', iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleMap = {
                'precision': ('Hierarchical Precision', '(HP)'),
                'recall': ('Hierarchical Recall', '(HR)'),
                'f1': ('Hierarchical F1', '(HF1)')
            }
            titleStr, typeStr = titleMap[metric]
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            s = self.eval[metric]
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            if metric in ['precision', 'scores']:
                # dimension of precision & scores: [TxRxAxM]
                s = s[:,:,aind,mind]
            else:
                # dimension of recall & f1: [TxAxM]
                s = s[:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((18,))
            stats[0] = _summarize('precision')
            stats[1] = _summarize('precision', iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize('precision', iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize('precision', areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize('precision', areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize('precision', areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize('recall', maxDets=self.params.maxDets[0])
            stats[7] = _summarize('recall', maxDets=self.params.maxDets[1])
            stats[8] = _summarize('recall', maxDets=self.params.maxDets[2])
            stats[9] = _summarize('recall', areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize('recall', areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize('recall', areaRng='large', maxDets=self.params.maxDets[2])
            stats[12] = _summarize('f1')
            stats[13] = _summarize('f1', iouThr=.5, maxDets=self.params.maxDets[2])
            stats[14] = _summarize('f1', iouThr=.75, maxDets=self.params.maxDets[2])
            stats[15] = _summarize('f1', areaRng='small', maxDets=self.params.maxDets[2])
            stats[16] = _summarize('f1', areaRng='medium', maxDets=self.params.maxDets[2])
            stats[17] = _summarize('f1', areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((15,))
            stats[0] = _summarize('precision', maxDets=20)
            stats[1] = _summarize('precision', maxDets=20, iouThr=.5)
            stats[2] = _summarize('precision', maxDets=20, iouThr=.75)
            stats[3] = _summarize('precision', maxDets=20, areaRng='medium')
            stats[4] = _summarize('precision', maxDets=20, areaRng='large')
            stats[5] = _summarize('recall', maxDets=20)
            stats[6] = _summarize('recall', maxDets=20, iouThr=.5)
            stats[7] = _summarize('recall', maxDets=20, iouThr=.75)
            stats[8] = _summarize('recall', maxDets=20, areaRng='medium')
            stats[9] = _summarize('recall', maxDets=20, areaRng='large')
            stats[10] = _summarize('f1', maxDets=20)
            stats[11] = _summarize('f1', maxDets=20, iouThr=.5)
            stats[12] = _summarize('f1', maxDets=20, iouThr=.75)
            stats[13] = _summarize('f1', maxDets=20, areaRng='medium')
            stats[14] = _summarize('f1', maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()


def hierarchical_prf_metric(
    hierarchy_tree: HierarchyTree,
    dt_label,
    gt_label,
    label_to_name: Dict = None,
    return_paths: bool = False
):
    """Compute hierarchical precision, recall, and F1 between predicted and GT labels.

    Optionally returns the raw paths for additional use in aggregation.
    """
    if label_to_name is not None:
        dt_label = label_to_name.get(int(dt_label), str(dt_label))
        gt_label = label_to_name.get(int(gt_label), str(gt_label))

    dt_path = hierarchy_tree.get_path(dt_label)
    gt_path = hierarchy_tree.get_path(gt_label)
    overlap = set(dt_path) & set(gt_path)
    hprecision = len(overlap) / len(dt_path)
    hrecall = len(overlap) / len(gt_path)
    hf1 = (2 * hprecision * hrecall / (hprecision + hrecall + 1e-6)) if (hprecision + hrecall) > 0 else 0.0
    results = {
        'hpr': hprecision,
        'hr': hrecall,
        'hf1': hf1,
    }
    if return_paths:
        results.update({
            'len_overlap': len(overlap),
            'len_dt': len(dt_path),
            'len_gt': len(gt_path),
            'pred_path': dt_path,
            'gt_path': gt_path,
        })
    return results
