import time
import copy
import datetime
from collections import defaultdict
from typing import Dict

import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm

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
        self._precompute_paths()

    def _precompute_paths(self):
        """Precomputes all paths in the hierarchy and stores them in a lookup table."""
        self.path_lookup = {}
        for name in self.hierarchy_tree.class_to_node.keys():
            # Store path as a set for fast intersection
            self.path_lookup[name] = set(self.hierarchy_tree.get_path(name))

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

        # Create a list of all evaluation tasks
        tasks = [(imgId, areaRng) for areaRng in p.areaRng for imgId in p.imgIds]

        # Use tqdm for a progress bar
        self.evalImgs = [evaluateImg(imgId, areaRng, maxDet)
                         for imgId, areaRng in tqdm(tasks, desc='Running per-image evaluation')]

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
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')[:maxDet]
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId][dtind, :] if len(self.ious[imgId]) > 0 else self.ious[imgId]
        ious = ious[:, gtind]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D), dtype=bool)
        # For "stealing", we need to track which detection index is matched to a GT
        gtm_idx = -np.ones((T, G), dtype=int)
        # We also need to store the hf1 and overlap of the current best match for each GT
        gt_hf1 = np.zeros((T, G))

        hierarchical_stats = {
            'overlaps': np.zeros((T, D)),
            'len_dt': np.zeros((D)),
            'len_gt': np.zeros((G)),
            'hprecision': np.zeros((T, D)),
            'hrecall': np.zeros((T, D))
        }
        # Precompute all path lengths
        for dind, d in enumerate(dt):
            dt_label = self.label_to_name.get(d['category_id'])
            if dt_label:
                hierarchical_stats['len_dt'][dind] = len(self.path_lookup.get(dt_label, []))
        for gind, g in enumerate(gt):
            gt_label = self.label_to_name.get(g['category_id'])
            if gt_label:
                hierarchical_stats['len_gt'][gind] = len(self.path_lookup.get(gt_label, []))


        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                # --- Non-Greedy Matching using Iterative Refinement ---
                
                # Keep track of which detections are currently matched
                d_matched = np.zeros(D, dtype=bool)
                
                # Use a list to manage detections that need to be matched or re-matched.
                # Initially, this is all detections, in confidence order.
                detections_to_process = list(range(D))

                # Safeguard against infinite loops
                loop_count = 0
                max_loops = D * D + 1 # Should be more than enough

                while detections_to_process:
                    if loop_count > max_loops:
                        print(f"WARNING: Breaking out of matching loop for img {imgId}, iouThr {t}. Exceeded max iterations.")
                        break
                    loop_count += 1

                    dind = detections_to_process.pop(0) # Get the next highest-confidence detection
                    d = dt[dind]

                    best_boost = 1e-9  # Use a small epsilon to ensure only real improvements cause a steal
                    best_gt_match_idx = -1
                    best_match_metrics = {}
                    best_iou = -1.0
                    
                    # For ancestor-ignoring: track best hf1 if no positive boost is found
                    best_unmatched_hf1 = -1.0
                    best_unmatched_gt_idx = -1

                    curr_d_label = self.label_to_name.get(d['category_id'])
                    if not curr_d_label:
                        continue

                    # Find the best possible GT match for the current detection
                    for gind, g in enumerate(gt):
                        # A GT is a potential candidate if IoU is sufficient
                        if ious[dind, gind] < t:
                            continue

                        gt_label = self.label_to_name.get(g['category_id'])
                        if not gt_label:
                            continue
                        
                        potential_match_metrics = hierarchical_prf_metric(
                            self.path_lookup, curr_d_label, gt_label, return_paths=True
                        )
                        potential_hf1 = potential_match_metrics['hf1']
                        
                        # The "boost" is the improvement over the GT's current match (if any)
                        existing_hf1 = gt_hf1[tind, gind]
                        boost = potential_hf1 - existing_hf1

                        if boost > best_boost:
                            best_boost = boost
                            best_gt_match_idx = gind
                            best_match_metrics = potential_match_metrics
                            best_iou = ious[dind, gind]
                        elif boost == best_boost and ious[dind, gind] > best_iou:
                            # Tie-break with IoU
                            best_gt_match_idx = gind
                            best_match_metrics = potential_match_metrics
                            best_iou = ious[dind, gind]
                        
                        # If no positive boost, track the best raw hf1 for potential ancestor-ignoring
                        if potential_hf1 > best_unmatched_hf1:
                            best_unmatched_hf1 = potential_hf1
                            best_unmatched_gt_idx = gind


                    # If a satisfactory match was found for this detection
                    if best_gt_match_idx != -1:
                        # Check if this GT was previously matched
                        prev_d_idx = gtm_idx[tind, best_gt_match_idx]

                        if prev_d_idx != -1:
                            # This is a "steal". The previously matched detection must be re-evaluated.
                            # Unset its previous match details
                            dtm[tind, prev_d_idx] = 0
                            d_matched[prev_d_idx] = False
                            hierarchical_stats['overlaps'][tind, prev_d_idx] = 0
                            hierarchical_stats['hprecision'][tind, prev_d_idx] = 0
                            hierarchical_stats['hrecall'][tind, prev_d_idx] = 0
                            # Add it back to the queue to find a new home.
                            detections_to_process.insert(0, prev_d_idx)

                        # Assign the new, better match
                        g = gt[best_gt_match_idx]
                        dtm[tind, dind] = g['id']
                        gtm[tind, best_gt_match_idx] = d['id']
                        gtm_idx[tind, best_gt_match_idx] = dind
                        d_matched[dind] = True
                        
                        # Store the metrics of the new best match
                        gt_hf1[tind, best_gt_match_idx] = best_match_metrics['hf1']
                        hierarchical_stats['overlaps'][tind, dind] = best_match_metrics['len_overlap']
                        hierarchical_stats['hprecision'][tind, dind] = best_match_metrics['hprecision']
                        hierarchical_stats['hrecall'][tind, dind] = best_match_metrics['hrecall']
                    
                    # --- Integrated Ancestor-Ignoring Logic ---
                    # If the detection couldn't find a match that offered a positive boost
                    else:
                        # Check if its best potential GT is already matched to a descendant
                        if best_unmatched_gt_idx != -1:
                            final_match_d_idx = gtm_idx[tind, best_unmatched_gt_idx]
                            if final_match_d_idx != -1: # If the GT is matched
                                final_match_d_label = self.label_to_name.get(dt[final_match_d_idx]['category_id'])
                                # If the existing match is a descendant of the current detection, ignore current
                                if (final_match_d_label and
                                    self.hierarchy_tree.is_descendant(final_match_d_label, curr_d_label)):
                                    dtIg[tind, dind] = True

        # --- Consolidated Ignore Logic ---
        # Find all GT IDs that are marked as ignore
        ignored_gt_ids = [g['id'] for g in gt if g['_ignore']]
        # Create a boolean mask where dtm has a match to an ignored GT.
        # This is True for every (tind, dind) where the matched GT ID is in our ignored list.
        matches_to_ignored_gt = np.isin(dtm, ignored_gt_ids)

        # Update dtIg: a detection is ignored if it was already marked for ignoring (e.g., ancestor logic)
        # OR if it's matched to an explicitly ignored GT.
        dtIg = np.logical_or(dtIg, matches_to_ignored_gt)

        # Vectorized check for unmatched detections outside the area range
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
        unmatched_and_outside_area = np.logical_and(dtm == 0, np.repeat(a, T, 0))
        dtIg = np.logical_or(dtIg, unmatched_and_outside_area)

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
        f1          = -np.ones((T,A,M))
        scores      = -np.ones((T,R,A,M))
        precision_soft = -np.ones((T, R, A, M))
        recall_soft    = -np.ones((T, A, M))
        f1_soft        = -np.ones((T, A, M))

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

                dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                
                # Hierarchical Stats
                overlaps   = np.concatenate([e['hstats']['overlaps'][:, :maxDet] for e in E], axis=1)[:, inds]
                hprecision = np.concatenate([e['hstats']['hprecision'][:, :maxDet] for e in E], axis=1)[:, inds]
                hrecall    = np.concatenate([e['hstats']['hrecall'][:, :maxDet] for e in E], axis=1)[:, inds]
                len_dt     = np.concatenate([e['hstats']['len_dt'][:maxDet]   for e in E], axis=0)[inds]
                len_gt     = np.concatenate([e['hstats']['len_gt'] for e in E], axis=0)

                # Mask ignored GTs
                valid_gt_mask = (gtIg == 0)  # shape: (G,)
                gt_lens_masked = len_gt[valid_gt_mask]  # keep only valid GTs

                # Now sum all nodes in valid GT paths (for each IoU threshold)
                npig = gt_lens_masked.sum(axis=0)
                n_gt = int((gtIg == 0).sum())
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
                    if nd:
                        f1_curve = 2 * pr * rc / (pr + rc + 1e-6)
                        f1[t, a, m] = np.max(f1_curve) if f1_curve.size > 0 else 0
                    else:
                        f1[t, a, m] = 0

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
                
                for t in range(T):
                    # 1) mask out ignored detections
                    valid_mask    = ~dtIg[t]                       # shape (D,)
                    soft_tp_precision_scores = hprecision[t][valid_mask]     # hPrecision in [0,1]
                    soft_tp_recall_scores = hrecall[t][valid_mask]           # hRecall in [0,1]
                    soft_fp_scores = 1.0 - soft_tp_precision_scores          # soft “false positives”

                    # 2) build running‐sum PR curve in classic micro style
                    cum_tp_precision = np.cumsum(soft_tp_precision_scores)   # shape (D_valid,)
                    cum_fp = np.cumsum(soft_fp_scores)
                    cum_tp_recall = np.cumsum(soft_tp_recall_scores)

                    # 3) compute recall & precision
                    rec_curve  = cum_tp_recall / (n_gt + np.spacing(1))
                    prec_curve = cum_tp_precision / (cum_tp_precision + cum_fp + np.spacing(1))

                    # 4) enforce monotonicity on precision
                    for i in range(len(prec_curve) - 2, -1, -1):
                        prec_curve[i] = max(prec_curve[i], prec_curve[i+1])

                    # 5) sample at your fixed recThrs grid
                    q = np.zeros((R,), dtype=float)
                    inds = np.searchsorted(rec_curve, p.recThrs, side='left')
                    for ri, pi in enumerate(inds):
                        if pi < len(prec_curve):
                            q[ri] = prec_curve[pi]
                    precision_soft[t, :, a, m] = q

                    # 6) record final recall & best‐F1
                    recall_soft[t, a, m] = rec_curve[-1] if rec_curve.size else 0.0
                    if rec_curve.size:
                        f1_scores = 2 * rec_curve * prec_curve / (rec_curve + prec_curve + 1e-6)
                        f1_soft[t, a, m] = f1_scores.max()
                    else:
                        f1_soft[t, a, m] = 0.0
                
        self.eval = {
            'params': p,
            'counts': [T, R, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision':      precision_soft,
            'recall':         recall_soft,
            'f1':             f1_soft,
            'scores':         scores,
            'precision_node': precision,
            'recall_node':    recall,
            'f1_node':        f1,
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
                'precision': ('Hierarchical Average Precision', '(HAP)'),
                'recall': ('Hierarchical Average Recall', '(HAR)'),
                'f1': ('Hierarchical Average F1', '(HAF1)'),
                'precision_node': ('Node-based Precision', '(Node-HAP)'),
                'recall_node': ('Node-based Recall', '(Node-HAR)'),
                'f1_node': ('Node-based F1', '(Node-HAF1)'),
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
            if metric in ['precision', 'precision_node', 'scores']:
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
            stats = np.zeros((36,))
            # Soft-binary (default) metrics
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
            # Node-based metrics (node = hard, ablation)
            stats[18] = _summarize('precision_node')
            stats[19] = _summarize('precision_node', iouThr=.5, maxDets=self.params.maxDets[2])
            stats[20] = _summarize('precision_node', iouThr=.75, maxDets=self.params.maxDets[2])
            stats[21] = _summarize('precision_node', areaRng='small', maxDets=self.params.maxDets[2])
            stats[22] = _summarize('precision_node', areaRng='medium', maxDets=self.params.maxDets[2])
            stats[23] = _summarize('precision_node', areaRng='large', maxDets=self.params.maxDets[2])
            stats[24] = _summarize('recall_node', maxDets=self.params.maxDets[0])
            stats[25] = _summarize('recall_node', maxDets=self.params.maxDets[1])
            stats[26] = _summarize('recall_node', maxDets=self.params.maxDets[2])
            stats[27] = _summarize('recall_node', areaRng='small', maxDets=self.params.maxDets[2])
            stats[28] = _summarize('recall_node', areaRng='medium', maxDets=self.params.maxDets[2])
            stats[29] = _summarize('recall_node', areaRng='large', maxDets=self.params.maxDets[2])
            stats[30] = _summarize('f1_node')
            stats[31] = _summarize('f1_node', iouThr=.5, maxDets=self.params.maxDets[2])
            stats[32] = _summarize('f1_node', iouThr=.75, maxDets=self.params.maxDets[2])
            stats[33] = _summarize('f1_node', areaRng='small', maxDets=self.params.maxDets[2])
            stats[34] = _summarize('f1_node', areaRng='medium', maxDets=self.params.maxDets[2])
            stats[35] = _summarize('f1_node', areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((30,))
            # Soft-binary (default) metrics
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
            # Node-based metrics
            stats[15] = _summarize('precision_node', maxDets=20)
            stats[16] = _summarize('precision_node', maxDets=20, iouThr=.5)
            stats[17] = _summarize('precision_node', maxDets=20, iouThr=.75)
            stats[18] = _summarize('precision_node', maxDets=20, areaRng='medium')
            stats[19] = _summarize('precision_node', maxDets=20, areaRng='large')
            stats[20] = _summarize('recall_node', maxDets=20)
            stats[21] = _summarize('recall_node', maxDets=20, iouThr=.5)
            stats[22] = _summarize('recall_node', maxDets=20, iouThr=.75)
            stats[23] = _summarize('recall_node', maxDets=20, areaRng='medium')
            stats[24] = _summarize('recall_node', maxDets=20, areaRng='large')
            stats[25] = _summarize('f1_node', maxDets=20)
            stats[26] = _summarize('f1_node', maxDets=20, iouThr=.5)
            stats[27] = _summarize('f1_node', maxDets=20, iouThr=.75)
            stats[28] = _summarize('f1_node', maxDets=20, areaRng='medium')
            stats[29] = _summarize('f1_node', maxDets=20, areaRng='large')
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
    path_lookup: Dict,
    dt_label,
    gt_label,
    return_paths: bool = False
):
    """Compute hierarchical precision, recall, and F1 between predicted and GT labels.

    Optionally returns the raw paths for additional use in aggregation.
    """
    dt_path = path_lookup.get(dt_label, set())
    gt_path = path_lookup.get(gt_label, set())

    len_dt_path = len(dt_path)
    len_gt_path = len(gt_path)

    overlap = dt_path & gt_path
    len_overlap = len(overlap)
    hprecision = len_overlap / len_dt_path if len_dt_path > 0 else 0.0
    hrecall = len_overlap / len_gt_path if len_gt_path > 0 else 0.0
    if hprecision + hrecall == 0:
        hf1 = 0.0
    else:
        hf1 = 2 * (hprecision * hrecall) / (hprecision + hrecall)

    if return_paths:
        return {
            'hf1': hf1,
            'hprecision': hprecision,
            'hrecall': hrecall,
            'len_overlap': len_overlap,
            'len_dt': len_dt_path,
            'len_gt': len_gt_path,
        }
    else:
        return hf1
