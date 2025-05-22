#!/usr/bin/env python
import argparse
import os
from pathlib import Path
from ray import tune, train, air # Added air for RunConfig
from ray.train.torch import TorchTrainer
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

from mmengine.config import Config
from hod.hpo import config_to_param_space
from hod.hpo.trainable import train_loop_per_worker
import pprint

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True,
                        help="Base MMEngine config path (relative to project root)")
    parser.add_argument("--num-samples", type=int, default=40,
                        help="Number of Ray Tune trials")
    parser.add_argument("--gpus-per-trial", type=float, default=1,
                        help="Number of GPUs to allocate for each trial. If > 1, DDP is used.")
    parser.add_argument("--cpus-per-worker", type=int, default=4,
                        help="Number of CPUs to allocate for each DDP worker within a trial.")
    args = parser.parse_args()

    print(f"DEBUG: tools/tune.py: Parsed arguments: args.gpus_per_trial = {args.gpus_per_trial}, args.num_samples = {args.num_samples}, args.cfg = {args.cfg}, args.cpus_per_worker = {args.cpus_per_worker}")
    try:
        print(f"DEBUG: tools/tune.py: Ray cluster resources: {ray.cluster_resources()}")
        print(f"DEBUG: tools/tune.py: Ray available resources: {ray.available_resources()}")
    except Exception as e:
        print(f"DEBUG: tools/tune.py: Could not get Ray cluster resources: {e}")

    abs_cfg_path = PROJECT_ROOT / args.cfg
    if not abs_cfg_path.exists():
        print(f"ERROR: Config file not found at {abs_cfg_path}.")
        return
    
    print(f"DEBUG: tools/tune.py: Attempting to load base config for param_space from: {abs_cfg_path}")
    base_cfg = Config.fromfile(str(abs_cfg_path))
    
    # This is the search space for the parameters that train_loop_per_worker will receive.
    # It includes shared HPO params (e.g., shared_decay) and model-specific choices.
    param_space_for_worker = config_to_param_space(base_cfg)

    print("DEBUG: param_space_for_worker (for train_loop_per_worker):")
    pprint.pprint(param_space_for_worker)

    # Initial train_loop_config with fixed parameters for train_loop_per_worker.
    # Tunable parameters from param_space_for_worker will be merged into this by Ray Tune.
    initial_train_loop_config = {
        "cfg_path": str(abs_cfg_path),
        "project_root_dir": str(PROJECT_ROOT),
        # Placeholder for other fixed params if any; HPO params come from param_space_for_worker
    }

    # For Tuner, the param_space needs to specify that we are tuning 'train_loop_config'
    # of the TorchTrainer.
    param_space_for_tuner = {
        "train_loop_config": param_space_for_worker
    }
    print("DEBUG: param_space_for_tuner (passed to tune.Tuner):")
    pprint.pprint(param_space_for_tuner)

    num_workers_for_trial = 1
    use_gpu_for_trial = False
    if args.gpus_per_trial > 0:
        use_gpu_for_trial = True
        if args.gpus_per_trial > 1:
            num_workers_for_trial = int(args.gpus_per_trial)
    
    print(f"DEBUG: tools/tune.py: TorchTrainer setup: num_workers_for_trial={num_workers_for_trial}, use_gpu_for_trial={use_gpu_for_trial}")

    torch_trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config=initial_train_loop_config, # Pass initial fixed config here
        scaling_config=train.ScalingConfig(
            num_workers=num_workers_for_trial,
            use_gpu=use_gpu_for_trial,
            resources_per_worker={"CPU": args.cpus_per_worker, "GPU": 1 if use_gpu_for_trial else 0}
        )
    )

    tuner = tune.Tuner(
        torch_trainer,
        param_space=param_space_for_tuner, # Pass the wrapped param_space
        tune_config=tune.TuneConfig(
            metric="mAP",
            mode="max",
            num_samples=args.num_samples,
            search_alg=OptunaSearch(metric="mAP", mode="max", seed=0),
            scheduler=ASHAScheduler() # metric/mode inherited from TuneConfig
        ),
        run_config=air.RunConfig( # Use air.RunConfig from ray.air
            name="hod_hpo_torchtrial",
            storage_path=str(PROJECT_ROOT / "work_dirs" / "ray_results"),
        )
    )
    
    results = tuner.fit()

    print("--- HPO Run Complete ---")
    if results.get_best_result():
        print("Best hyperparameters found were: ", results.get_best_result().config)
        print("Best mAP: ", results.get_best_result().metrics.get("mAP"))
    else:
        print("No best result found. The experiment might have failed or was interrupted.")

if __name__ == "__main__":
    main()