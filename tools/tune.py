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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_hpo_config(config_path: str) -> ConfigDict:
    """Loads HPO settings from a Python file."""
    abs_config_path = PROJECT_ROOT / config_path
    if not abs_config_path.exists():
        raise FileNotFoundError(f"HPO config file not found at {abs_config_path}")
    
    cfg = Config.fromfile(str(abs_config_path))
    return cfg._cfg_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True,
                        help="Base MMEngine config path for model/dataset (relative to project root)")
    parser.add_argument("--hpo-cfg", required=True,
                        help="HPO settings config path (relative to project root)")
    parser.add_argument("--gpus-per-trial", type=float, default=1,
                        help="Number of GPUs to allocate for each trial. If > 1, DDP is used.")
    parser.add_argument("--cpus-per-worker", type=int, default=4,
                        help="Number of CPUs to allocate for each DDP worker within a trial.")
    args = parser.parse_args()

    hpo_settings = load_hpo_config(args.hpo_cfg)

    num_samples = hpo_settings.get("num_samples", 10)
    metric = hpo_settings.get("metric", "mAP") 
    mode = hpo_settings.get("mode", "max")
    search_alg_config = hpo_settings.get("search_alg")
    scheduler_config = hpo_settings.get("scheduler")

    abs_cfg_path = PROJECT_ROOT / args.cfg
    if not abs_cfg_path.exists():
        print(f"ERROR: Config file not found at {abs_cfg_path}.")
        return
    
    base_cfg = Config.fromfile(str(abs_cfg_path))
    param_space_for_worker = config_to_param_space(base_cfg._cfg_dict)

    initial_train_loop_config = {
        "cfg_path": str(abs_cfg_path),
        "project_root_dir": str(PROJECT_ROOT),
    }

    param_space_for_tuner = {
        "train_loop_config": param_space_for_worker
    }

    num_workers_for_trial = 1
    use_gpu_for_trial = False
    if args.gpus_per_trial > 0:
        use_gpu_for_trial = True
        if args.gpus_per_trial > 1:
            num_workers_for_trial = int(args.gpus_per_trial)
    
    torch_trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config=initial_train_loop_config, 
        scaling_config=train.ScalingConfig(
            num_workers=num_workers_for_trial,
            use_gpu=use_gpu_for_trial,
            resources_per_worker={"CPU": args.cpus_per_worker, "GPU": 1 if use_gpu_for_trial else 0}
        )
    )

    search_alg = None
    if search_alg_config:
        alg_type = search_alg_config.get("type")
        alg_params = search_alg_config.get("params", {})
        if alg_type == "OptunaSearch":
            search_alg = OptunaSearch(metric=metric, mode=mode, **alg_params)
        elif alg_type == "HyperOptSearch":
            search_alg = HyperOptSearch(metric=metric, mode=mode, **alg_params)
        elif alg_type == "BasicVariantGenerator":
            search_alg = BasicVariantGenerator(**alg_params)
        else:
            print(f"Warning: Unknown search_alg type: {alg_type}. No search algorithm will be used.")
    
    if not search_alg: 
        search_alg = OptunaSearch(metric=metric, mode=mode, seed=0)

    scheduler = None
    if scheduler_config:
        sch_type = scheduler_config.get("type")
        sch_params = scheduler_config.get("params", {})
        if sch_type == "ASHAScheduler":
            scheduler = ASHAScheduler(**sch_params)
        elif sch_type == "HyperBandScheduler":
            scheduler = HyperBandScheduler(**sch_params)
        elif sch_type == "PopulationBasedTraining":
            scheduler = PopulationBasedTraining(metric=metric, mode=mode, **sch_params)
        elif sch_type == "None" or sch_type is None:
            scheduler = None
        else:
            print(f"Warning: Unknown scheduler type: {sch_type}. No scheduler will be used.")
            scheduler = None 
            
    if not scheduler and scheduler_config and scheduler_config.get("type") not in ["None", None]:
        scheduler = ASHAScheduler()
    elif scheduler_config is None: 
        scheduler = ASHAScheduler()

    tuner = tune.Tuner(
        torch_trainer,  # type: ignore[arg-type]
        param_space=param_space_for_tuner,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=num_samples
        ),
        run_config=tune.RunConfig(
            name="hod_hpo_torchtrial",
            storage_path=str(PROJECT_ROOT / "work_dirs" / "ray_results"),
        )
    )
    
    results = tuner.fit()

    print("--- HPO Run Complete ---")
    best_result_obj = results.get_best_result()
    if best_result_obj:
        print("Best hyperparameters found were: ", best_result_obj.config)
        if best_result_obj.metrics:
            print(f"Best {metric}: ", best_result_obj.metrics.get(metric))
        else:
            print("Best result found, but no metrics were reported.")
    else:
        print("No best result found. The experiment might have failed or was interrupted.")

if __name__ == "__main__":
    main()
