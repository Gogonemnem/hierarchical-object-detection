"""
hod.hpo
~~~~~~~
Light-weight hyper-parameter optimisation helpers for MMEngine projects.
"""
from .auto_space import config_to_param_space
from .trainable import train_trial

__all__ = [
    "Choice", "LogUniform", "RandInt", "RandFloat",
    "config_to_param_space", "train_trial",
]
