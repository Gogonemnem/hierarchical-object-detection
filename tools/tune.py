#!/usr/bin/env python
import argparse
import os
from pathlib import Path
from ray import tune, train
from ray.train.torch import TorchTrainer
# Import necessary searchers and schedulers
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler, PopulationBasedTraining

from mmengine.config import Config, ConfigDict # Ensure ConfigDict is imported
from hod.hpo import config_to_param_space
from hod.hpo.trainable import train_loop_per_worker
import pprint

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_hpo_config(config_path: str) -> ConfigDict:
    """Loads HPO settings from a Python file."""
    abs_config_path = PROJECT_ROOT / config_path
    if not abs_config_path.exists():
        raise FileNotFoundError(f"HPO config file not found at {abs_config_path}")
    
    # Use mmengine.Config to load the Python file as a module
    # This allows defining Python objects directly in the config
    cfg = Config.fromfile(str(abs_config_path))
    return cfg._cfg_dict # Return as a ConfigDict (which behaves like a dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True,
                        help="Base MMEngine config path for model/dataset (relative to project root)")
    parser.add_argument("--hpo-cfg", required=True,
                        help="HPO settings config path (relative to project root)")
    # num-samples is now expected to be in hpo_cfg
    # parser.add_argument("--num-samples", type=int, default=40,
    #                     help="Number of Ray Tune trials")
    parser.add_argument("--gpus-per-trial", type=float, default=1,
                        help="Number of GPUs to allocate for each trial. If > 1, DDP is used.")
    parser.add_argument("--cpus-per-worker", type=int, default=4,
                        help="Number of CPUs to allocate for each DDP worker within a trial.")
    args = parser.parse_args()

    print(f"DEBUG: tools/tune.py: Parsed arguments: args.gpus_per_trial = {args.gpus_per_trial}, args.cfg = {args.cfg}, args.hpo_cfg = {args.hpo_cfg}, args.cpus_per_worker = {args.cpus_per_worker}")

    # Load HPO configuration
    hpo_settings = load_hpo_config(args.hpo_cfg)
    print("DEBUG: tools/tune.py: Loaded HPO settings:")
    pprint.pprint(hpo_settings)

    num_samples = hpo_settings.get("num_samples", 10) # Default if not in HPO config
    metric = hpo_settings.get("metric", "mAP")
    mode = hpo_settings.get("mode", "max")
    search_alg_config = hpo_settings.get("search_alg")
    scheduler_config = hpo_settings.get("scheduler")

    try:
        import ray
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
    param_space_for_worker = config_to_param_space(base_cfg._cfg_dict)

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

    # Instantiate search algorithm based on hpo_settings
    search_alg = None
    if search_alg_config:
        alg_type = search_alg_config.get("type")
        alg_params = search_alg_config.get("params", {})
        if alg_type == "OptunaSearch":
            search_alg = OptunaSearch(metric=metric, mode=mode, **alg_params)
        elif alg_type == "HyperOptSearch":
            # Note: HyperOptSearch might require param_space to be defined using hp.choice etc.
            search_alg = HyperOptSearch(metric=metric, mode=mode, **alg_params)
        elif alg_type == "BasicVariantGenerator": # For random search
            search_alg = BasicVariantGenerator(**alg_params)
        # Add other search algorithms as needed
        else:
            print(f"Warning: Unknown search_alg type: {alg_type}. No search algorithm will be used.")
    
    if not search_alg: # Default if not specified or unknown
        print("Info: Using default OptunaSearch.")
        search_alg = OptunaSearch(metric=metric, mode=mode, seed=0)


    # Instantiate scheduler based on hpo_settings
    scheduler = None
    if scheduler_config:
        sch_type = scheduler_config.get("type")
        sch_params = scheduler_config.get("params", {})
        if sch_type == "ASHAScheduler":
            # metric and mode are inherited from TuneConfig
            scheduler = ASHAScheduler(**sch_params)
        elif sch_type == "HyperBandScheduler":
            # metric and mode are inherited from TuneConfig
            scheduler = HyperBandScheduler(**sch_params)
        elif sch_type == "PopulationBasedTraining":
            # PBT requires more specific setup, e.g., hyperparam_mutations
            # Ensure sch_params includes these if using PBT
            # PBT typically requires metric and mode directly.
            scheduler = PopulationBasedTraining(metric=metric, mode=mode, **sch_params)
        elif sch_type == "None" or sch_type is None:
            scheduler = None
        else:
            print(f"Warning: Unknown scheduler type: {sch_type}. No scheduler will be used.")
            scheduler = None # Explicitly None
            
    if not scheduler and scheduler_config and scheduler_config.get("type") not in ["None", None]:
        print(f"Info: Scheduler type '{scheduler_config.get('type')}' specified but not recognized or failed to init. Defaulting to ASHAScheduler.")
        # Default ASHAScheduler will also inherit metric and mode from TuneConfig
        scheduler = ASHAScheduler()
    elif scheduler_config is None: # If no scheduler config block, use default
        print("Info: No scheduler configuration found. Using default ASHAScheduler.")
        # Default ASHAScheduler will also inherit metric and mode from TuneConfig
        scheduler = ASHAScheduler()


    tuner = tune.Tuner(
        torch_trainer,
        param_space=param_space_for_tuner,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=num_samples
        ),
        run_config=tune.RunConfig( # Use train.RunConfig
            name="hod_hpo_torchtrial",
            storage_path=str(PROJECT_ROOT / "work_dirs" / "ray_results"),
            # checkpoint_config can be added here if needed
        )
    )
    
    results = tuner.fit()

    print("--- HPO Run Complete ---")
    best_result_obj = results.get_best_result()
    if best_result_obj:
        print("Best hyperparameters found were: ", best_result_obj.config)
        if best_result_obj.metrics:
            print("Best mAP: ", best_result_obj.metrics.get("mAP"))
        else:
            print("Best result found, but no metrics were reported.")
    else:
        print("No best result found. The experiment might have failed or was interrupted.")

if __name__ == "__main__":
    main()