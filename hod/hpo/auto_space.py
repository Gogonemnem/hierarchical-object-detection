"""
Convert special 'sampling spec' dictionaries in an MMEngine Config into
a flat dictionary of Ray Tune samplers, which is then unflattened to a
nested param_space. Complex choices are flattened for compatibility.
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence, Dict, List, Union
from ray import tune
from mmengine.config import Config

# Factory for creating Ray Tune samplers from stubs
_SAMPLER_FACTORY: Dict[str, Any] = {
    "RandInt":    lambda s: tune.randint(s["lower"], s["upper"] + 1), # Ray's randint is inclusive upper
    "RandFloat":  lambda s: tune.uniform(s["lower"], s["upper"]),
    "Uniform":    lambda s: tune.uniform(s["lower"], s["upper"]),
    "LogUniform": lambda s: tune.loguniform(s["lower"], s["upper"]),
    # "Choice" is handled specially
}

def _is_sampler_stub(node: Any) -> bool:
    """Checks if a node is a dictionary representing a sampler stub."""
    return isinstance(node, Mapping) and "type" in node and \
           (node["type"] in _SAMPLER_FACTORY or node["type"] == "Choice")

def _resolve_node_to_sampler_or_structure(node_config: Any) -> Any:
    """
    Recursively resolves a configuration node into its final form for a tune.choice list.
    Sampler stubs are converted to Ray Tune sampler objects.
    Dicts/lists are traversed, and their contents are recursively resolved.
    Literals are returned as is.
    """
    if _is_sampler_stub(node_config):
        sampler_type = node_config["type"]
        if sampler_type == "Choice":
            # Recursively resolve options for this inner choice
            inner_options = [_resolve_node_to_sampler_or_structure(opt) for opt in node_config.get("options", [])]
            return tune.choice(inner_options)
        else: # RandInt, Uniform, etc.
            return _SAMPLER_FACTORY[sampler_type](node_config)
    elif isinstance(node_config, Mapping):
        return {k: _resolve_node_to_sampler_or_structure(v) for k, v in node_config.items()}
    elif isinstance(node_config, Sequence) and not isinstance(node_config, (str, bytes)):
        return [_resolve_node_to_sampler_or_structure(item) for item in node_config]
    else: # Literal
        return node_config

def _recursive_build_flat_search_space(
    config_node: Any,
    current_path_parts: List[str],
    flat_search_space: Dict[str, Any]
) -> None:
    """
    Recursively traverses the config dict.
    - Simple sampler stubs are added to flat_search_space with a dot-separated key.
    - Complex choices (where options are dicts defining components with more stubs) are flattened.
    - Other dicts/lists are traversed.
    """
    current_key_str = ".".join(current_path_parts)

    if _is_sampler_stub(config_node):
        sampler_type = config_node["type"]

        if sampler_type == "Choice":
            options = config_node.get("options", [])
            
            # Determine if this is a choice between user-defined components (complex)
            # or a choice between literals/direct-samplers/nested-structures (simple).
            is_choice_of_components = False
            if options:
                def is_potential_component_dict(opt_node: Any) -> bool:
                    # A component dict has a 'type' key that isn't a known sampler type.
                    return isinstance(opt_node, Mapping) and "type" in opt_node and \
                           opt_node["type"] not in _SAMPLER_FACTORY and opt_node["type"] != "Choice"

                if any(is_potential_component_dict(opt) for opt in options):
                    if all(is_potential_component_dict(opt) for opt in options):
                        is_choice_of_components = True
                    else:
                        # Mixed options (e.g., component dict and a literal/sampler).
                        # Treat as a simple choice; _resolve_node_to_sampler_or_structure will handle recursion.
                        print(f"Warning: Choice at '{current_key_str}' has mixed component dicts and other option types. Treating as a simple choice.")
                        is_choice_of_components = False
            
            if is_choice_of_components:
                # Complex Choice: This choice selects between different component configurations.
                # Flatten parameters of each component into the main flat_search_space.
                # Example: param_space will have 'my_choice._CHOICE_TYPE_' and 'my_choice.ComponentA.param1', 'my_choice.ComponentB.paramX'.
                choice_selector_key = current_key_str + "._CHOICE_TYPE_"
                option_type_identifiers = []

                for opt_component_stub in options: # Known to be component dicts here
                    component_type_name = opt_component_stub["type"]
                    option_type_identifiers.append(component_type_name)

                    # Recursively define/flatten parameters for this specific component type,
                    # namespacing them under the component_type_name.
                    for param_key, param_value_stub in opt_component_stub.items():
                        if param_key == "type":  # Already used as the identifier
                            continue
                        
                        param_path_parts = current_path_parts + [component_type_name, param_key]
                        # Delegate to the main recursive function to handle this parameter value.
                        # This will add samplers or literals at '...component_type_name.param_key'.
                        _recursive_build_flat_search_space(param_value_stub, param_path_parts, flat_search_space)
                
                if option_type_identifiers: 
                    flat_search_space[choice_selector_key] = tune.choice(list(set(option_type_identifiers)))
                # If options was empty, option_type_identifiers will be empty, nothing added. Correct.

            else: # Simple Choice: tune.choice([resolved_option1, resolved_option2, ...])
                  # Options can be literals, sampler stubs, or nested dicts/lists containing them.
                  # Each option needs to be fully resolved (samplers instantiated).
                resolved_options = [_resolve_node_to_sampler_or_structure(opt) for opt in options]
                if resolved_options or not options: # tune.choice([]) is valid if options list was empty
                    flat_search_space[current_key_str] = tune.choice(resolved_options)
        
        else: # RandInt, Uniform, LogUniform, etc. (already handled by _SAMPLER_FACTORY)
            flat_search_space[current_key_str] = _SAMPLER_FACTORY[sampler_type](config_node)

    elif isinstance(config_node, Mapping):
        # Regular dictionary, recurse over its key-value pairs
        for key, value_node in config_node.items():
            _recursive_build_flat_search_space(value_node, current_path_parts + [key], flat_search_space)
    elif isinstance(config_node, Sequence) and not isinstance(config_node, (str, bytes)):
        # List or tuple.
        # If it contains sampler stubs, they should ideally be part of a 'Choice' or handled
        # if a list itself is a hyperparameter (e.g. tune.choice([[1,2],[3,4]])).
        # Here, we recurse on elements to handle nested structures within lists if they occur,
        # primarily for cases where lists contain further dicts with stubs.
        for i, item_node in enumerate(config_node):
            # Path for list items includes index, e.g., key.0, key.1
            _recursive_build_flat_search_space(item_node, current_path_parts + [str(i)], flat_search_space)
    # Else: it's a literal value (e.g., int, float, str, simple list/dict without stubs).
    # These are not part of the search space themselves, so they are ignored here.
    # They will be part of the base config loaded in train_trial.

def config_to_param_space(cfg_or_dict: Union[Config, Mapping]) -> Dict[str, Any]: # Adjusted type hint
    """
    Converts an MMEngine Config object or a dictionary containing hyperparameter
    stubs into a nested dictionary suitable for Ray Tune's `param_space`.
    Shared HPO parameters defined in `hpo_shared_params` are processed first.
    Sampler stubs are converted to Ray Tune sampler objects.
    Complex choices (options are dicts with nested stubs) are flattened
    to be more compatible with search algorithms like Optuna.
    """
    # Ensure input is a dictionary
    if isinstance(cfg_or_dict, Config): # More specific type check for MMEngine Config
        config_dict = cfg_or_dict.to_dict()
    elif isinstance(cfg_or_dict, dict):
        config_dict = cfg_or_dict.copy() # Use a copy to avoid modifying the original
    else:
        raise TypeError("Input must be an MMEngine Config object or a dictionary.")

    flat_search_space: Dict[str, Any] = {}

    # 1. Process hpo_shared_params first
    hpo_shared_params_config = config_dict.pop('hpo_shared_params', {})
    for name, sampler_stub_dict in hpo_shared_params_config.items():
        if _is_sampler_stub(sampler_stub_dict) and sampler_stub_dict["type"] != "Choice":
            flat_search_space[name] = _SAMPLER_FACTORY[sampler_stub_dict["type"]](sampler_stub_dict)
        else:
            # Potentially handle error or warning: shared params should be direct samplers, not choices.
            print(f"Warning: Shared HPO parameter '{name}' is not a simple sampler stub. It will be ignored.")
            
    # 2. Process the rest of the configuration
    _recursive_build_flat_search_space(config_dict, [], flat_search_space)
    
    # Unflatten the search space from dot-separated keys to a nested dict structure.
    # Ray Tune expects param_space to be a nested dict mirroring the config structure
    # where the leaves are the sampler objects.
    nested_param_space = {}
    for key_path_str, sampler_object in flat_search_space.items():
        parts = key_path_str.split('.')
        current_level = nested_param_space
        for i, part in enumerate(parts):
            if i == len(parts) - 1: # Last part is the key for the sampler
                current_level[part] = sampler_object
            else:
                # Ensure intermediate dictionaries exist
                current_level = current_level.setdefault(part, {})
                # This simple unflattening creates dicts for list indices (e.g., {'0': ...}).
                # _finalize_config_choices in trainable.py needs to be aware if it traverses
                # such paths, but typically hyperparameters are in dict structures.
    
    return nested_param_space

# Example usage (for testing this file directly):
if __name__ == '__main__':
    import pprint
    example_config_stub = {
        'model': {
            'type': 'MyModel',
            'bbox_head': {
                'loss_cls': { # Example of a non-choice param
                    'type': 'FocalLoss', 
                    'gamma': {'type': 'Uniform', 'lower': 1.0, 'upper': 3.0},
                    'alpha': 0.25 # Literal
                },
                'loss_embed': { # Complex choice
                    'type': 'Choice',
                    'options': [
                        {
                            'type': 'HierarchicalContrastiveLoss', # Option 1 type
                            'loss_weight': {'type': 'LogUniform', 'lower': 0.01, 'upper': 1.0},
                            'ann_file': 'path/to/ann1.json', 
                            'temperature': {'type': 'Uniform', 'lower': 0.05, 'upper': 0.2}
                        },
                        {
                            'type': 'EntailmentConeLoss', # Option 2 type
                            'beta': {'type': 'Uniform', 'lower': 0.1, 'upper': 0.5},
                            'loss_weight': 1.0 
                        }
                    ]
                }
            }
        },
        'optimizer': { # Another non-choice param
            'lr': {'type': 'LogUniform', 'lower': 1e-5, 'upper': 1e-3}
        },
        'data': { # Simple choice
            'samples_per_gpu': {'type': 'Choice', 'options': [2, 4, 8]} 
        },
        'other_choice': { # Choice of direct samplers
             'type': 'Choice',
             'options': [
                 {'type': 'Uniform', 'lower': 0, 'upper': 1},
                 {'type': 'RandInt', 'lower': 10, 'upper': 20}
             ]
        }
    }

    param_space = config_to_param_space(example_config_stub)
    print("Generated param_space (flattening strategy):")
    pprint.pprint(param_space)
    # Expected param_space structure for the 'loss_embed' part (after unflattening):
    # 'model': {
    #   'bbox_head': {
    #     'loss_embed': {
    #         '_CHOICE_TYPE_': <ray.tune.sample.Categorical object for ['HierarchicalContrastiveLoss', 'EntailmentConeLoss']>,
    #         'HierarchicalContrastiveLoss': { # Parameters for this choice
    #             'loss_weight': <ray.tune.sample.LogUniform object ...>,
    #             'ann_file': 'path/to/ann1.json', # Literal carried through
    #             'temperature': <ray.tune.sample.Float object ...>
    #         },
    #         'EntailmentConeLoss': { # Parameters for this choice
    #             'beta': <ray.tune.sample.Float object ...>,
    #             'loss_weight': 1.0 # Literal carried through
    #         }
    #     }
    #   }
    # }
    #
    # Explanation:
    # Literals (e.g., 'ann_file', 'loss_weight: 1.0') from within a chosen component's definition
    # in the HPO config are included in the `param_space` as fixed values (not samplers).
    # Ray Tune then includes these literals in the `trial_config` that is passed to
    # `train_loop_per_worker` (as `current_trial_hyperparameters`).
    # The `_finalize_config_choices` function in `hod.hpo.trainable` uses these literals
    # when reconstructing the configuration for the chosen component. It merges these
    # HPO-defined parameters (both sampled and literal) with the original base model config.
    # This ensures that non-tuned parameters of a chosen component are correctly set according
    # to the HPO configuration stub for that component.
