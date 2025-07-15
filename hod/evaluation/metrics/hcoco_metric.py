import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict

from mmengine.fileio import load
from mmengine.logging import MMLogger

from mmdet.datasets.api_wrappers import COCO
from mmdet.evaluation.metrics.coco_metric import CocoMetric
from mmdet.registry import METRICS

from hod.datasets.api_wrappers.hierarchical_coco import HierarchicalCOCOeval

@METRICS.register_module()
class HierarchicalCocoMetric(CocoMetric):
    def __init__(self,
                 ann_file: str = '',
                 *args,
                 **kwargs):
        super().__init__(*args, ann_file=ann_file, **kwargs)
        self.classwise = False
        self.taxonomy: dict = load(ann_file).get('taxonomy', {})

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            # TODO: May refactor fast_eval_recall to an independent metric?
            # fast eval recall
            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    preds, self.proposal_nums, self.iou_thrs, logger=logger)
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                logger.info(log_msg)
                continue

            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error(
                    'The testing results of the whole dataset is empty.')
                break

            # if self.use_mp_eval:
            #     coco_eval = COCOevalMP(self._coco_api, coco_dt, iou_type)
            # else:
            coco_eval = HierarchicalCOCOeval(self.taxonomy, self._coco_api, coco_dt, iou_type)

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            # mapping of hcocoEval.stats
            coco_metric_names = {
                'hmAP': 0, 'hmAP_50': 1, 'hmAP_75': 2, 'hmAP_s': 3, 'hmAP_m': 4, 'hmAP_l': 5,
                'hAR@100': 6, 'hAR@300': 7, 'hAR@1000': 8, 'hAR_s@1000': 9, 'hAR_m@1000': 10, 'hAR_l@1000': 11,
                'hAF1': 12, 'hAF1_50': 13, 'hAF1_75': 14, 'hAF1_s': 15, 'hAF1_m': 16, 'hAF1_l': 17,
                # Node-based (hard) metrics
                'hmAP_node': 18, 'hmAP_50_node': 19, 'hmAP_75_node': 20, 'hmAP_s_node': 21, 'hmAP_m_node': 22, 'hmAP_l_node': 23,
                'hAR@100_node': 24, 'hAR@300_node': 25, 'hAR@1000_node': 26, 'hAR_s@1000_node': 27, 'hAR_m@1000_node': 28, 'hAR_l@1000_node': 29,
                'hAF1_node': 30, 'hAF1_50_node': 31, 'hAF1_75_node': 32, 'hAF1_s_node': 33, 'hAF1_m_node': 34, 'hAF1_l_node': 35,
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            if metric == 'proposal':
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if metric_items is None:
                    metric_items = [
                        'hAR@100', 'hAR@300', 'hAR@1000', 'hAR_s@1000',
                        'hAR_m@1000', 'hAR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                if metric_items is None:
                    metric_items = [
                        'hmAP', 'hmAP_50', 'hmAP_75', 'hmAP_s', 'hmAP_m', 'hmAP_l',
                        'hAR@100', 'hAR@300', 'hAR@1000', 'hAR_s@1000', 'hAR_m@1000', 'hAR_l@1000',
                        'hAF1', 'hAF1_50', 'hAF1_75', 'hAF1_s', 'hAF1_m', 'hAF1_l',
                        # Node-based (hard) metrics
                        'hmAP_node', 'hmAP_50_node', 'hmAP_75_node', 'hmAP_s_node', 'hmAP_m_node', 'hmAP_l_node',
                        'hAR@100_node', 'hAR@300_node', 'hAR@1000_node', 'hAR_s@1000_node', 'hAR_m@1000_node', 'hAR_l@1000_node',
                        'hAF1_node', 'hAF1_50_node', 'hAF1_75_node', 'hAF1_s_node', 'hAF1_m_node', 'hAF1_l_node',
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = coco_eval.stats[coco_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')

                stat_groups = {
                    'hmAP': coco_eval.stats[0:6],
                    'hAR': coco_eval.stats[6:12],
                    'hAF1': coco_eval.stats[12:18],
                    # Node-based (hard) metrics
                    'hmAP_node': coco_eval.stats[18:24],
                    'hAR_node': coco_eval.stats[24:30],
                    'hAF1_node': coco_eval.stats[30:36],
                }

                for label, values in stat_groups.items():
                    values_str = ' '.join(f'{v:.3f}' for v in values)
                    logger.info(f'{metric}_{label}_copypaste: {values_str}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
