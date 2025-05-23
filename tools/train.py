# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

# Import the pre-training function
# Assuming it will be moved to hod.training.prototypes
# If it stays in tools.pretrain_prototypes, the import path will need adjustment
from hod.training.prototypes import perform_prototype_pretraining
from pathlib import Path # Add Path import

# Patch torch.load to disable `weights_only=True` introduced in PyTorch 2.6
# This avoids UnpicklingError when resuming from checkpoints saved with full objects.
# Safe to use ONLY when loading checkpoints from trusted sources (e.g., your own training).
import torch
_real_load = torch.load

def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False  # allow loading full objects (e.g., mmengine ConfigDict, HistoryBuffer)
    return _real_load(*args, **kwargs)

torch.load = safe_load

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # --- Prototype Pre-training Integration ---
    pretrain_cfg = cfg.get('prototype_pretrain_cfg')

    if pretrain_cfg and pretrain_cfg.get('enable', False):
        print("Prototype pre-training enabled via configuration.")
        
        # Get pre-training parameters from config, with defaults
        epochs = pretrain_cfg.get('epochs', 100)
        device_override = pretrain_cfg.get('device') # Can be None
        force_pretrain = pretrain_cfg.get('force_pretrain', False)
        # output_dir_name = pretrain_cfg.get('output_subdir', 'prototype_pretrain') # More configurable output
        # output_filename = pretrain_cfg.get('output_filename', 'pretrained_prototypes.pth')

        pretrain_output_dir = Path(cfg.work_dir) / 'prototype_pretrain' # Keeping it simple for now
        pretrain_output_file = pretrain_output_dir / 'pretrained_prototypes.pth'
        pretrain_output_dir.mkdir(parents=True, exist_ok=True)

        if not force_pretrain and pretrain_output_file.exists():
            print(f"Found existing pre-trained prototype checkpoint: {pretrain_output_file}")
            print("Skipping pre-training. Set prototype_pretrain_cfg.force_pretrain=True in config to override.")
            cfg.load_from = str(pretrain_output_file)
            cfg.resume = False # Ensure we are not resuming if we load this pre-trained model
        else:
            if force_pretrain and pretrain_output_file.exists():
                print(f"force_pretrain=True. Re-running prototype pre-training and overwriting {pretrain_output_file}")
            elif not pretrain_output_file.exists():
                print(f"No existing pre-trained checkpoint found at {pretrain_output_file}. Starting pre-training.")
            
            pretrain_device = device_override
            if pretrain_device is None:
                pretrain_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            print(f"Starting prototype pre-training for {epochs} epochs on device '{pretrain_device}'.")
            print(f"Pre-trained checkpoint will be saved to: {pretrain_output_file}")

            try:
                saved_checkpoint_path = perform_prototype_pretraining(
                    model_config_path=args.config, # Main config path
                    output_checkpoint_file=pretrain_output_file,
                    epochs=epochs,
                    device=pretrain_device
                )
                print(f"Prototype pre-training complete. Checkpoint saved to: {saved_checkpoint_path}")
                cfg.load_from = str(saved_checkpoint_path)
                cfg.resume = False
            except Exception as e:
                print(f"Error during prototype pre-training: {e}")
                raise RuntimeError(f"Prototype pre-training failed: {e}")
        
        print(f"Set 'cfg.load_from' to '{cfg.load_from}' for main training.")
    # --- End of Prototype Pre-training Integration ---

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # Workaround for reentrant backward issues with Swin/DDP
    try:
        runner.model._set_static_graph()
    except AttributeError:
        pass
    
    # start training
    runner.train()


if __name__ == '__main__':
    main()
