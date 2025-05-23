# HPO Configuration File

# --- General ---
num_samples = 1  # Number of trials Ray Tune will run
metric = "val/coco/bbox_mF1"    # Metric to optimize
mode = "max"      # "max" or "min"

# --- Search Algorithm ---
# Available types: "OptunaSearch", "HyperOptSearch", "BasicVariantGenerator" (for random)
# Refer to Ray Tune documentation for parameters for each search algorithm.
search_alg = dict(
    type="OptunaSearch",
    params=dict(
        seed=42,
        # Example: Define a specific sampler for Optuna
        # sampler=dict(type="TPESampler", params=dict(seed=123, consider_prior=True)),
        # Example: Define a specific pruner for Optuna
        # pruner=dict(type="MedianPruner", params=dict(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)),
    )
)

# --- Scheduler ---
# Available types: "ASHAScheduler", "HyperBandScheduler", "PopulationBasedTraining", "None"
# Refer to Ray Tune documentation for parameters for each scheduler.
scheduler = dict(
    type="ASHAScheduler",
    params=dict(
        # These are often inherited from TuneConfig (metric, mode) if not specified here,
        # but can be overridden.
        # time_attr="training_iteration", # or "time_total_s" or other reported metric
        # max_t=100, # Max iterations/epochs/time per trial
        # grace_period=10, # Minimum iterations/epochs/time before stopping
        # reduction_factor=2 # Factor for reducing resources/trials
    )
)

# --- Example: HyperOptSearch ---
# search_alg = dict(
#     type="HyperOptSearch",
#     params=dict(
#         random_state_seed=123,
#         # For HyperOpt, you might need to define the search space differently
#         # if not using Ray Tune's automatic conversion.
#         # n_initial_points=10  # Number of initial random samples for HyperOpt
#     )
# )

# --- Example: Random Search (using BasicVariantGenerator) ---
# search_alg = dict(
#     type="BasicVariantGenerator",
#     params=dict(
#         random_state=777,
#         # max_concurrent=4 # Limits number of concurrent trials for random search if needed
#     )
# )

# --- Example: HyperBandScheduler ---
# scheduler = dict(
#     type="HyperBandScheduler",
#     params=dict(
#         # time_attr="training_iteration",
#         # max_t=81,
#         # reduction_factor=3
#     )
# )

# --- Example: No Scheduler ---
# scheduler = dict(type="None")

# --- Example: PopulationBasedTraining (PBT) ---
# Note: PBT requires `hyperparam_mutations` to be defined,
# which should correspond to your `param_space` in `tools/tune.py`.
# scheduler = dict(
#     type="PopulationBasedTraining",
#     params=dict(
#         time_attr="training_iteration",
#         # metric="episode_reward_mean", # PBT often uses its own metric
#         # mode="max",
#         perturbation_interval=60, # In seconds or iterations, depending on time_attr
#         # hyperparam_mutations = {
#         #     # This needs to mirror the structure of your param_space in tune.py
#         #     # specifically the 'train_loop_config' part.
#         #     "train_loop_config": {
#         #         "model": {
#         #             "bbox_head": {
#         #                 "loss_cls": {
#         #                     # Assuming 'alpha' is a hyperparameter you're tuning
#         #                     "alpha": tune.uniform(0.1, 0.9),
#         #                 },
#         #                 "loss_bbox": {
#         #                     # Assuming 'loss_weight' is a hyperparameter
#         #                     "loss_weight": tune.loguniform(0.5, 2.0),
#         #                 }
#         #             }
#         #         },
#         #         # Example for a top-level hyperparameter if you have one
#         #         # "learning_rate": tune.loguniform(1e-4, 1e-1),
#         #     }
#         # }
#         # # You might also need to specify custom_explore_fn for more complex mutations
#     )
# )
