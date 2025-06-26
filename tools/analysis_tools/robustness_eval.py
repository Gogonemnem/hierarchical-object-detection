# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from argparse import ArgumentParser

import numpy as np
from mmengine.fileio import load


def get_coco_style_results(filename,
                           task='bbox',
                           metric=None,
                           prints='mPC',
                           aggregate='benchmark'):

    assert aggregate in ['benchmark', 'all']

    if prints == 'all':
        prints = ['P', 'mPC', 'rPC']
    elif isinstance(prints, str):
        prints = [prints]
    for p in prints:
        assert p in ['P', 'mPC', 'rPC']

    eval_output = load(filename)

    # Check for empty results early to prevent errors
    if not eval_output or not any(eval_output.values()):
        print(f"Warning: No results found in file: {filename}")
        return {}

    # Infer metrics from the first result entry to make the script adaptable
    # to any metrics present in the results file (like HP, HR, HF1).
    first_result = next(iter(next(iter(eval_output.values())).values()))
    all_keys = list(first_result.keys())

    # Filter for the current task (e.g., 'bbox') and create a clean list
    # of metric names like 'bbox_mAP', 'bbox_mF1', etc.
    metrics = sorted([k.split('/')[-1] for k in all_keys if task in k])

    if not metrics:
        print(
            f"Warning: No metrics found for task '{task}' in results file: {filename}"
        )
        return {}

    num_distortions = len(list(eval_output.keys()))
    # Dynamically determine the number of severities from the results file
    max_severity = 0
    for distortion in eval_output:
        if eval_output[distortion]:
            max_severity = max(max_severity, *eval_output[distortion].keys())
    num_severities = max_severity + 1
    results = np.zeros((num_distortions, num_severities, len(metrics)), dtype='float32')

    for corr_i, distortion in enumerate(eval_output):
        for severity in eval_output[distortion]:
            metric_dict = eval_output[distortion][severity]

            # Create a clean dict: {'bbox_mAP': 0.588, ...}
            clean_metric_dict = {
                k.split('/')[-1]: v
                for k, v in metric_dict.items()
            }

            for metric_j, metric_name in enumerate(metrics):
                if metric_name in clean_metric_dict:
                    results[corr_i, severity,
                            metric_j] = clean_metric_dict[metric_name]

    # Severity 0 is the clean performance, same for all corruptions
    P = results[0, 0, :]
    # print results shape for debugging
    print(f'Results shape: {results.shape}')
    if aggregate == 'benchmark':
        # benchmark corruptions are the first 15
        mPC = np.mean(results[:15, 1:, :], axis=(0, 1))
    else:
        mPC = np.mean(results[:, 1:, :], axis=(0, 1))

    # Use np.divide to handle division by zero safely, fixing the warning
    rPC = np.divide(mPC, P, out=np.zeros_like(mPC), where=P != 0)

    print(f'\nmodel: {osp.basename(filename)}')
    print(f'task: {task}')

    # Generic printing for all found metrics
    if 'P' in prints:
        print('\n--- Performance on Clean Data [P] ---')
        for metric_i, metric_name in enumerate(metrics):
            print(f'{metric_name:25} = {P[metric_i]:.4f}')

    if 'mPC' in prints:
        print('\n--- Mean Performance under Corruption [mPC] ---')
        for metric_i, metric_name in enumerate(metrics):
            print(f'{metric_name:25} = {mPC[metric_i]:.4f}')

    if 'rPC' in prints:
        print('\n--- Relative Performance under Corruption [rPC] ---')
        for metric_i, metric_name in enumerate(metrics):
            print(f'{metric_name:25} = {rPC[metric_i]:.4f}')

    all_results = {
        'P': dict(zip(metrics, P)),
        'mPC': dict(zip(metrics, mPC)),
        'rPC': dict(zip(metrics, rPC))
    }
    return all_results


def get_voc_style_results(filename, prints='mPC', aggregate='benchmark'):

    assert aggregate in ['benchmark', 'all']

    if prints == 'all':
        prints = ['P', 'mPC', 'rPC']
    elif isinstance(prints, str):
        prints = [prints]
    for p in prints:
        assert p in ['P', 'mPC', 'rPC']

    eval_output = load(filename)

    if not eval_output:
        print(f"Warning: No results found in file: {filename}")
        return np.array([])

    num_distortions = len(list(eval_output.keys()))
    # Dynamically determine the number of severities from the results file
    max_severity = 0
    for distortion in eval_output:
        if eval_output[distortion]:
            max_severity = max(max_severity, *eval_output[distortion].keys())
    num_severities = max_severity + 1
    results = np.zeros((num_distortions, num_severities, 20), dtype='float32')

    for i, distortion in enumerate(eval_output):
        for severity in eval_output[distortion]:
            mAP = [
                eval_output[distortion][severity][j]['ap']
                for j in range(len(eval_output[distortion][severity]))
            ]
            results[i, severity, :] = mAP

    P = results[0, 0, :]
    if aggregate == 'benchmark':
        mPC = np.mean(results[:15, 1:, :], axis=(0, 1))
    else:
        mPC = np.mean(results[:, 1:, :], axis=(0, 1))
    rPC = mPC / P

    print(f'\nmodel: {osp.basename(filename)}')
    if 'P' in prints:
        print(f'Performance on Clean Data [P] in AP50 = {np.mean(P):0.3f}')
    if 'mPC' in prints:
        print('Mean Performance under Corruption [mPC] in AP50 = '
              f'{np.mean(mPC):0.3f}')
    if 'rPC' in prints:
        print('Relative Performance under Corruption [rPC] in % = '
              f'{np.mean(rPC) * 100:0.1f}')

    return np.mean(results, axis=2, keepdims=True)


def get_results(filename,
                dataset='coco',
                task='bbox',
                metric=None,
                prints='mPC',
                aggregate='benchmark'):
    assert dataset in ['coco', 'voc', 'cityscapes']

    if dataset in ['coco', 'cityscapes']:
        results = get_coco_style_results(
            filename,
            task=task,
            metric=metric,
            prints=prints,
            aggregate=aggregate)
    elif dataset == 'voc':
        if task != 'bbox':
            print('Only bbox analysis is supported for Pascal VOC')
            print('Will report bbox results\n')
        if metric not in [None, ['AP'], ['AP50']]:
            print('Only the AP50 metric is supported for Pascal VOC')
            print('Will report AP50 metric\n')
        results = get_voc_style_results(
            filename, prints=prints, aggregate=aggregate)

    return results


def get_distortions_from_file(filename):

    eval_output = load(filename)

    return get_distortions_from_results(eval_output)


def get_distortions_from_results(eval_output):
    distortions = []
    for i, distortion in enumerate(eval_output):
        distortions.append(distortion.replace('_', ' '))
    return distortions


def main():
    parser = ArgumentParser(description='Corruption Result Analysis')
    parser.add_argument('filename', help='result file path')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['coco', 'voc', 'cityscapes'],
        default='coco',
        help='dataset type')
    parser.add_argument(
        '--task',
        type=str,
        nargs='+',
        choices=['bbox', 'segm'],
        default=['bbox'],
        help='task to report')
    parser.add_argument(
        '--metric',
        nargs='+',
        choices=[
            None, 'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10',
            'AR100', 'ARs', 'ARm', 'ARl'
        ],
        default=None,
        help='metric to report')
    parser.add_argument(
        '--prints',
        type=str,
        nargs='+',
        choices=['P', 'mPC', 'rPC'],
        default='mPC',
        help='corruption benchmark metric to print')
    parser.add_argument(
        '--aggregate',
        type=str,
        choices=['all', 'benchmark'],
        default='benchmark',
        help='aggregate all results or only those \
        for benchmark corruptions')

    args = parser.parse_args()

    for task in args.task:
        get_results(
            args.filename,
            dataset=args.dataset,
            task=task,
            metric=args.metric,
            prints=args.prints,
            aggregate=args.aggregate)


if __name__ == '__main__':
    main()