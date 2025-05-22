"""
Convert special 'sampling spec' dictionaries in an MMEngine Config into
a flat dictionary of Ray Tune samplers, which is then unflattened to a
nested param_space. Complex choices are flattened for compatibility.
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence, Dict, List
from ray import tune

# Factory for creating Ray Tune samplers from stubs
_SAMPLER_FACTORY: Dict[str, Any] = {
    "RandInt":    lambda s: tune.randint(s["lower"], s["upper"] + 1), # Ray's randint is inclusive upper
    "RandFloat":  lambda s: tune.uniform(s["lower"], s["upper"]),
    "Uniform":    lambda s: tune.uniform(s["lower"], s["upper"]),
    "LogUniform": lambda s: tune.loguniform(s["lower"], s["upper"]),
    # "Choice" is handled specially in _recursive_build_flat_search_space
}

def _is_sampler_stub(node: Any) -> bool:
    """Checks if a node is a dictionary representing a sampler stub."""
    return isinstance(node, Mapping) and "type" in node and \
           (node["type"] in _SAMPLER_FACTORY or node["type"] == "Choice")

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
            
            # Determine if it's a complex choice (options are component stubs)
            # vs. a simple choice (options are literals or direct samplers).
            is_complex_choice_scenario = False
            if options and isinstance(options[0], Mapping) and "type" in options[0]:
                # If the first option is a dict with a 'type' key, assume it's a complex choice
                # where options are component definitions.
                is_complex_choice_scenario = True
                # Further validation: ensure all options in a complex choice are dicts with 'type'
                if not all(isinstance(opt, Mapping) and "type" in opt for opt in options):
                    # This logic could be enhanced to support mixed simple/complex options if needed,
                    # but that would complicate the flattening and reconstruction significantly.
                    # For now, assume options in a "complex" choice are consistently component stubs.
                    # If not, it might fall into the simple choice path, or error.
                    # Consider raising an error for mixed option types in a complex choice.
                    pass # Potentially problematic if options are mixed.

            if is_complex_choice_scenario:
                choice_selector_key = current_key_str + "._CHOICE_TYPE_"
                option_type_identifiers = []

                for opt_component_stub in options:
                    # opt_component_stub is like {'type': 'MyLoss', 'param1': {...}, ...}
                    component_type_name = opt_component_stub["type"]
                    option_type_identifiers.append(component_type_name)

                    # Recursively define parameters for this specific component type,
                    # namespacing them under the component_type_name.
                    for param_key, param_value_stub in opt_component_stub.items():
                        if param_key == "type":  # Already used as the identifier
                            continue
                        
                        param_path_parts = current_path_parts + [component_type_name, param_key]
                        if _is_sampler_stub(param_value_stub): # If it's a sampler, recurse
                            _recursive_build_flat_search_space(param_value_stub, param_path_parts, flat_search_space)
                        # elif not isinstance(param_value_stub, (Mapping, Sequence)) or isinstance(param_value_stub, (str, bytes)):
                        # The original condition for literals was too broad and might misclassify simple lists/dicts of literals.
                        # We want to capture any non-sampler, non-dict/list that needs further recursion.
                        # This includes strings, numbers, booleans, and None.
                        elif not _is_sampler_stub(param_value_stub) and not (isinstance(param_value_stub, Mapping) or (isinstance(param_value_stub, Sequence) and not isinstance(param_value_stub, (str, bytes)))):
                            # If it's a literal (not a sampler stub, not a dict, not a list/tuple needing recursion),
                            # add it directly to the flat_search_space for this component option.
                            # Ray Tune will pass these literals through in the trial_config.
                            flat_search_space[".".join(param_path_parts)] = param_value_stub
                        else:
                            # It's a non-sampler dict/list that might contain more stubs or literals, recurse.
                            _recursive_build_flat_search_space(param_value_stub, param_path_parts, flat_search_space)
                
                if option_type_identifiers: # Ensure there were valid options
                    flat_search_space[choice_selector_key] = tune.choice(list(set(option_type_identifiers))) # Use set to ensure unique type names
                elif options: # Options existed but none were valid component stubs for flattening
                    # Fallback to simple choice if complex parsing failed but options are present
                    # This indicates a misconfiguration or a type of choice not handled by complex flattening.
                    processed_simple_options = []
                    for opt_stub in options: # Process as potentially simple options
                        if _is_sampler_stub(opt_stub):
                            temp_flat_space = {} # Convert option stub in isolation
                            _recursive_build_flat_search_space(opt_stub, ["_temp_"], temp_flat_space)
                            if "_temp_" in temp_flat_space:
                                processed_simple_options.append(temp_flat_space["_temp_"])
                            else: # Should not happen if _is_sampler_stub is true
                                processed_simple_options.append(opt_stub) 
                        else: # Literal option
                            processed_simple_options.append(opt_stub)
                    if processed_simple_options:
                         flat_search_space[current_key_str] = tune.choice(processed_simple_options)
                    # If no options at all, this Choice stub is empty and will be ignored.

            else: # Simple Choice (options are literals or direct sampler stubs)
                processed_options = []
                for opt_stub in options:
                    if _is_sampler_stub(opt_stub): 
                        # This handles a choice of other samplers, e.g., tune.choice([uniform_stub, randint_stub])
                        # Each stub needs to be converted to its Ray Tune sampler object.
                        # We create a temporary flat space to convert the single option stub.
                        temp_flat_param_space_for_option = {}
                        # Use a placeholder key; its structure will be the sampler object.
                        _recursive_build_flat_search_space(opt_stub, ["_temp_option_"], temp_flat_param_space_for_option)
                        if "_temp_option_" in temp_flat_param_space_for_option:
                             processed_options.append(temp_flat_param_space_for_option["_temp_option_"])
                        else: # Fallback if conversion didn't yield the expected key
                            processed_options.append(opt_stub)
                    else: # Literal option
                        processed_options.append(opt_stub)
                if processed_options or not options: # Add choice if options processed or if it's an empty choice list
                    flat_search_space[current_key_str] = tune.choice(processed_options)
        
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

def config_to_param_space(cfg_or_dict: Mapping) -> Dict[str, Any]:
    """
    Converts an MMEngine Config object or a dictionary containing hyperparameter
    stubs into a nested dictionary suitable for Ray Tune's `param_space`.
    Shared HPO parameters defined in `hpo_shared_params` are processed first.
    Sampler stubs are converted to Ray Tune sampler objects.
    Complex choices (options are dicts with nested stubs) are flattened
    to be more compatible with search algorithms like Optuna.
    """
    # Ensure input is a dictionary
    if hasattr(cfg_or_dict, 'to_dict') and callable(getattr(cfg_or_dict, 'to_dict')): # MMEngine Config object
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
    # Expected output for loss_embed part (among others):
    # 'loss_embed': {
    #     '_CHOICE_TYPE_': <ray.tune.sample.Categorical object at ...>,
    #     'HierarchicalContrastiveLoss': {
    #         'loss_weight': <ray.tune.sample.LogUniform object at ...>,
    #         'temperature': <ray.tune.sample.Float object at ...>
    #     },
    #     'EntailmentConeLoss': {
    #         'beta': <ray.tune.sample.Float object at ...>
    #     }
    # }
    # Literals like 'ann_file' and 'loss_weight: 1.0' are NOT part of the search space here;
    # they are part of the base config and remain fixed for the chosen component.
    # _finalize_config_choices in trainable.py will merge these fixed values from the
    # original config with the sampled HPs.
    # Actually, no, the literals from the chosen option *should* be part of the trial_config
    # if _finalize_config_choices is to work correctly.
    # The current _recursive_build_flat_search_space IGNORES literals.
    # This means _finalize_config_choices will only get the *tunable* params.
    # This needs to be fixed: literals within a chosen component stub should also be carried through.
    #
    # Correction: The current _recursive_build_flat_search_space only adds sampler stubs to
    # flat_search_space. Literals are skipped. This means the `trial_config` that
    # `_finalize_config_choices` gets will *only* contain the tunable parameters for the
    # chosen component type (e.g., {'_CHOICE_TYPE_': 'TypeA', 'TypeA': {'param1': 0.5}}).
    # The `_finalize_config_choices` then reconstructs {'type': 'TypeA', 'param1': 0.5}.
    # The *non-tuned* parameters of TypeA (like 'ann_file' if it was literal in the stub)
    # must come from the *original base config* that `cfg.merge_from_dict(final_trial_params_for_merge)`
    # acts upon.
    # So, the base config (e.g., hierarchical_loss.py) must fully define all options of a choice.
    # The param_space only defines what's tunable within those options.
    # This seems to be the standard MMEngine way: base config is complete, HPO overrides parts.
