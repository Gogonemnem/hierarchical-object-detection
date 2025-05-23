import pprint
from pathlib import Path
import os
from typing import Any, Dict, List, Union, Optional, Tuple # Added Optional, Tuple
import copy

from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner
from ray import train
from ray.air import session

def _get_original_option_definition(base_choice_node: Union[Dict[str, Any], ConfigDict], chosen_type_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    """
    Finds the original definition and its index for a chosen_type_name within a base 'Choice' node's options.
    Returns (option_definition_dict, index_in_options_list) or (None, None) if not found.
    """
    options_list = base_choice_node.get("options")
    if isinstance(options_list, list):
        for i, opt_stub in enumerate(options_list):
            if isinstance(opt_stub, dict) and opt_stub.get("type") == chosen_type_name:
                return opt_stub, i
    return None, None

def _finalize_config_choices(
    current_trial_config_node: Union[Dict[str, Any], List[Any], Any], 
    current_base_cfg_node: Optional[Union[Dict[str, Any], List[Any], Any]], 
    current_path_keys: List[str], 
    choice_paths_accumulator: List[List[str]]
) -> Union[Dict[str, Any], List[Any], Any]:
    """
    Recursively processes a node from the HPO-generated trial configuration 
    (`current_trial_config_node`) against the corresponding node from the original 
    base configuration (`current_base_cfg_node`).

    Args:
        current_trial_config_node: The current node (dict, list, or literal) 
            from the trial's hyperparameter configuration.
        current_base_cfg_node: The corresponding node from the base MMEngine 
            configuration. Can be `None` if the trial config introduces a structure 
            not present in the base (e.g., parameters for a chosen option that had 
            no detailed definition in the base 'Choice' stub).
        current_path_keys: A list of keys representing the path taken to reach 
            the `current_base_cfg_node` from the root of the base configuration. 
            This is used to identify 'Choice' nodes for later deletion.
        choice_paths_accumulator: A list that accumulates the `current_path_keys` 
            for each 'Choice' node that is successfully resolved. These paths 
            point to the original 'Choice' stubs in the base configuration that 
            need to be removed before merging the resolved choice.

    Returns:
        A new dictionary, list, or literal representing the resolved configuration 
        for the current node, ready to be incorporated into the final configuration 
        for the trial.

    Key Operations:
    1.  **'Choice' Resolution**: If `current_base_cfg_node` is a 'Choice' (identified 
        by `{'type': 'Choice', ...}`) and `current_trial_config_node` contains 
        `'_CHOICE_TYPE_'` (indicating which option was selected by HPO), this function:
        a.  Constructs the resolved MMEngine component (e.g., `{'type': 'SelectedTypeName', ...params}`).
        b.  Uses `_get_original_option_definition` to find the original definition 
            of the selected option within the `current_base_cfg_node`'s 'options'.
        c.  Recursively calls itself to process the parameters for the selected option, 
            merging HPO-provided values with any defaults or nested structures from 
            the original option's definition.
        d.  Adds `current_path_keys` to `choice_paths_accumulator` to mark the original 
            'Choice' stub in the base config for deletion.
    2.  **List Traversal**: If `current_trial_config_node` is a list, it iterates through 
        its items, recursively calling itself for each item, attempting to find a 
        corresponding item in `current_base_cfg_node` if it's also a list.
    3.  **Dictionary Traversal**: If `current_trial_config_node` is a dictionary (and not 
        a 'Choice' resolution scenario as above), it iterates through its key-value 
        pairs, recursively calling itself. It attempts to find corresponding child 
        nodes in `current_base_cfg_node`.
        - Includes logic to filter out keys from `current_trial_config_node` that might 
          represent unchosen branches of a 'Choice' if the HPO parameter space includes 
          all potential branches (e.g., if `current_base_cfg_node` is a 'Choice' but 
          `current_trial_config_node` didn't have `_CHOICE_TYPE_`).
    4.  **Literal Handling**: If `current_trial_config_node` is a literal (int, str, 
        float, bool, None), it's returned as is.
    """

    # Case 1: Current node from trial_config is a list
    if isinstance(current_trial_config_node, list):
        new_list = []
        for i, trial_item in enumerate(current_trial_config_node):
            base_item = None
            if isinstance(current_base_cfg_node, list) and i < len(current_base_cfg_node):
                base_item = current_base_cfg_node[i]
            # Path for list items: if base_item exists, path is to it. If not, path is conceptual.
            new_path_for_recursion = current_path_keys + [str(i)]
            new_list.append(_finalize_config_choices(trial_item, base_item, new_path_for_recursion, choice_paths_accumulator))
        return new_list

    # Case 2: Current node from trial_config is not a dictionary (hence a literal)
    if not isinstance(current_trial_config_node, dict):
        return current_trial_config_node # Return literal as is

    # Case 3: Current node from trial_config is a dictionary.
    # This is where 'Choice' resolution or further dictionary traversal happens.
    if isinstance(current_base_cfg_node, (dict, ConfigDict)) and \
       current_base_cfg_node.get('type') == 'Choice' and \
       '_CHOICE_TYPE_' in current_trial_config_node:
        
        # This block handles the resolution of a 'Choice' defined in current_base_cfg_node.
        if current_path_keys: 
            choice_paths_accumulator.append(list(current_path_keys)) # Record path to this Choice

        chosen_type_name = current_trial_config_node['_CHOICE_TYPE_']
        # Parameters for the chosen type from HPO (e.g., current_trial_config_node['LPIPSLoss'])
        params_for_chosen_type = current_trial_config_node.get(chosen_type_name, {})
        
        final_resolved_node = {'type': chosen_type_name} # Start building the MMEngine compatible node
        
        # Attempt to find the original definition of the chosen option in the base config's 'Choice'
        original_option_definition_in_base_cfg, option_idx_in_base = _get_original_option_definition(current_base_cfg_node, chosen_type_name)

        if original_option_definition_in_base_cfg is None:
            # The chosen type (e.g., 'LPIPSLoss') was selected by HPO but was NOT explicitly
            # listed as an option in the base_cfg's 'Choice' node.
            # In this case, we only use the parameters provided by HPO for this type.
            # Recursively process these HPO params without a direct base_cfg counterpart for them.
            processed_params = {}
            if isinstance(params_for_chosen_type, dict):
                for p_key, p_value in params_for_chosen_type.items():
                    # The path for recursion is somewhat "artificial" for the base_cfg side here,
                    # as these params don't map to a pre-existing structure within an option
                    # of the base 'Choice'. It's constructed based on the trial_config structure.
                    artificial_path_for_recursion = current_path_keys + [chosen_type_name, p_key]
                    processed_params[p_key] = _finalize_config_choices(
                        p_value,
                        None, # No corresponding base config for these specific HPO-defined params
                        artificial_path_for_recursion, # Path mainly for consistent recursive calls
                        choice_paths_accumulator # Accumulator is passed through
                    )
            final_resolved_node.update(processed_params)
            return final_resolved_node

        # The chosen type WAS found as an option in the base_cfg's 'Choice' node.
        # Now, merge HPO parameters with the original definition of that option.
        if isinstance(params_for_chosen_type, dict):
            processed_params = {}
            for p_key, p_value in params_for_chosen_type.items():
                # Get the specific parameter's original config from the base option's definition
                original_param_cfg_in_base_option = original_option_definition_in_base_cfg.get(p_key)
                
                # Path to the parameter's definition within the base configuration's chosen option.
                # This is crucial if p_value itself is a nested structure that might contain another 'Choice'.
                # Example: base_cfg.model.head.loss -> (Choice Node at current_path_keys)
                #          options: [ {type: 'LossA', paramX: {...}}, {type: 'LossB', paramY: 10} ]
                # If 'LossA' is chosen, and HPO gives params for 'paramX',
                # path_to_base_param_definition will be current_path_keys + ['options', '0' (if LossA is 0th option), 'paramX']
                path_to_base_param_definition = current_path_keys + ['options', str(option_idx_in_base), p_key]
                
                processed_params[p_key] = _finalize_config_choices(
                    p_value, # The HPO value for this parameter
                    original_param_cfg_in_base_option, # The base config for this parameter (could be None or a complex dict)
                    path_to_base_param_definition, # Accurate path in base_cfg for further recursion
                    choice_paths_accumulator 
                )
            final_resolved_node.update(processed_params)
            
        return final_resolved_node
    else:
        # This branch handles:
        # 1. Regular dictionary traversal: current_trial_config_node is a dict, but
        #    current_base_cfg_node is not a 'Choice' or is not being resolved as such here
        #    (e.g., trial_config_node lacks '_CHOICE_TYPE_' for a base 'Choice').
        # 2. It also serves as a path for parts of the trial_config that don't directly
        #    correspond to a 'Choice' being resolved at this level of recursion.
        new_dict_for_merging = {}
        for key, trial_value_node in current_trial_config_node.items():
            # Filter: If current_base_cfg_node IS a 'Choice' (but we are in this 'else' branch,
            # meaning trial_config_node didn't have `_CHOICE_TYPE_` to resolve it in the block above),
            # we need to be careful not to process keys from trial_config_node that are merely
            # names of *other* (unchosen) options from that base 'Choice'. This can happen if
            # the HPO parameter space (from auto_space.py) includes all possible branches.
            if isinstance(current_base_cfg_node, (dict, ConfigDict)) and \
               current_base_cfg_node.get('type') == 'Choice': 
                # trial_config_node.get('_CHOICE_TYPE_') would be None or not the key we are iterating on.
                chosen_type_in_trial_node_if_any = current_trial_config_node.get('_CHOICE_TYPE_')
                
                # We skip this 'key' from trial_config_node if:
                #   a) It's not the '_CHOICE_TYPE_' key itself.
                #   b) It's not the key corresponding to the (hypothetically, if present) chosen type.
                #   c) AND it IS the name of one of the types listed in the base 'Choice' node's options.
                if key != chosen_type_in_trial_node_if_any and key != '_CHOICE_TYPE_':
                    is_an_option_type_name_in_base_choice = False
                    base_options = current_base_cfg_node.get("options")
                    if isinstance(base_options, list):
                        for opt_stub in base_options:
                            if isinstance(opt_stub, dict) and opt_stub.get("type") == key:
                                is_an_option_type_name_in_base_choice = True
                                break
                    if is_an_option_type_name_in_base_choice:
                        continue # Skip this key, it's an unchosen option's branch in trial_config
            
            base_cfg_child_node = None
            # Safely get the child from base_cfg, if base_cfg is a dict-like structure
            if isinstance(current_base_cfg_node, (dict, ConfigDict, Config)): 
                base_cfg_child_node = current_base_cfg_node.get(key)

            new_path_for_recursion = current_path_keys + [key]
            new_dict_for_merging[key] = _finalize_config_choices(
                trial_value_node, 
                base_cfg_child_node, 
                new_path_for_recursion, 
                choice_paths_accumulator
            )
        return new_dict_for_merging

def _delete_node_by_path(root_config_node: Union[Config, ConfigDict, dict], path_keys: List[str]) -> bool:
    """
    Deletes a node from a configuration structure (Config, ConfigDict, or dict)
    based on a list of path keys.
    Returns True if deletion was successful or path already not found, False on error during deletion.
    """
    if not path_keys:
        return True # No path, nothing to delete

    current_parent_for_access = root_config_node
    # parent_for_deletion will hold the actual dict/ConfigDict from which to delete
    parent_for_deletion: Union[ConfigDict, dict, None] = None 
    key_to_delete_in_parent = None

    for i, key_segment in enumerate(path_keys):
        is_last_segment = (i == len(path_keys) - 1)
        
        # Determine the actual container to look into (e.g., _cfg_dict for Config)
        if isinstance(current_parent_for_access, Config):
            # If current_parent_for_access is Config, we operate on its _cfg_dict
            # For the last segment, parent_for_deletion becomes this _cfg_dict
            # For traversal, next_node is fetched from _cfg_dict, current_parent_for_access becomes next_node
            container_to_check = current_parent_for_access._cfg_dict
        elif isinstance(current_parent_for_access, (ConfigDict, dict)):
            container_to_check = current_parent_for_access
        else: # Cannot traverse further if not Config, ConfigDict, or dict
            return True # Path segment not traversable, effectively node doesn't exist as expected

        if key_segment not in container_to_check:
            return True # Path segment not found, nothing to delete at this specific path

        if is_last_segment:
            parent_for_deletion = container_to_check # This is the actual dict/ConfigDict
            key_to_delete_in_parent = key_segment
            break
        
        next_node = container_to_check[key_segment]
        # Check if the next node is suitable for further traversal
        if not isinstance(next_node, (Config, ConfigDict, dict)):
            return True # Cannot traverse to the full depth, effectively node doesn't exist as expected
        current_parent_for_access = next_node

    if parent_for_deletion is not None and key_to_delete_in_parent is not None:
        try:
            del parent_for_deletion[key_to_delete_in_parent]
            return True
        except Exception as e:
            return False # Actual error during deletion
    
    # If parent_for_deletion or key_to_delete_in_parent is None here, it means the loop didn't correctly identify the target for deletion
    # (e.g., empty path_keys, though handled, or path led to a non-dict before the end).
    # However, previous checks should catch most of these, returning True (as "nothing to delete there").
    # This return is a fallback or for unexpected loop termination.
    return True # Default to True if no deletion error occurred and loop completed (e.g. path didn't fully resolve to a deletable item)

def _substitute_shared_params_in_config_obj(config_node: Any, trial_config: Dict[str, Any]):
    if isinstance(config_node, (Config, ConfigDict)): # MMEngine Config or ConfigDict
        for key in list(config_node.keys()): # Iterate over a copy of keys for safe modification
            item = config_node[key]
            if isinstance(item, str) and item.startswith("hpo_ref:"):
                ref_key = item.split(":", 1)[1]
                if ref_key in trial_config:
                    config_node[key] = trial_config[ref_key]
            elif isinstance(item, (Config, ConfigDict, dict, list)):
                _substitute_shared_params_in_config_obj(item, trial_config)
    elif isinstance(config_node, dict): # Standard Python dict
        for key in list(config_node.keys()):
            item = config_node[key]
            if isinstance(item, str) and item.startswith("hpo_ref:"):
                ref_key = item.split(":", 1)[1]
                if ref_key in trial_config:
                    config_node[key] = trial_config[ref_key]
            elif isinstance(item, (Config, ConfigDict, dict, list)):
                _substitute_shared_params_in_config_obj(item, trial_config)
    elif isinstance(config_node, list): # Standard Python list
        for i in range(len(config_node)):
            item = config_node[i]
            if isinstance(item, str) and item.startswith("hpo_ref:"):
                ref_key = item.split(":", 1)[1]
                if ref_key in trial_config:
                    config_node[i] = trial_config[ref_key]
            elif isinstance(item, (Config, ConfigDict, dict, list)):
                _substitute_shared_params_in_config_obj(item, trial_config)

def train_loop_per_worker(current_trial_hyperparameters: dict):
    cfg_path = current_trial_hyperparameters.pop("cfg_path")
    project_root_dir = current_trial_hyperparameters.pop("project_root_dir")

    os.chdir(project_root_dir)

    cfg = Config.fromfile(cfg_path)
    
    choice_paths_resolved_in_cfg: List[List[str]] = []

    final_trial_params_for_merge_raw = _finalize_config_choices(
        copy.deepcopy(current_trial_hyperparameters), 
        cfg,                         
        [],                          
        choice_paths_resolved_in_cfg 
    )
    
    # Ensure final_trial_params_for_merge_raw is a dict for merge_from_dict
    if not isinstance(final_trial_params_for_merge_raw, dict):
        # This should ideally not happen if current_trial_hyperparameters is a dict.
        # If it occurs, it implies a fundamental issue in how choices are resolved or structured.
        # For now, we'll raise an error or log heavily, as merging a non-dict is problematic.
        print(f"CRITICAL ERROR (Rank {train.get_context().get_world_rank()}): final_trial_params_for_merge is not a dict, it is {type(final_trial_params_for_merge_raw)}. This will likely cause a crash. Params: {final_trial_params_for_merge_raw}")
        # Fallback to an empty dict to prevent immediate crash, but this is not a real fix.
        final_trial_params_for_merge = {}
    else:
        final_trial_params_for_merge = final_trial_params_for_merge_raw

    for path_keys_to_delete in choice_paths_resolved_in_cfg:
        if not path_keys_to_delete: # Should not happen if _finalize_config_choices appends non-empty paths
            continue
        if not _delete_node_by_path(cfg, path_keys_to_delete):
            # This indicates an actual error occurred during the deletion attempt for an existing path.
            # The CRITICAL ERROR for non-dict merge_raw is more severe, this is a config cleanup issue.
            print(f"ERROR (Rank {train.get_context().get_world_rank()}): Failed to delete original Choice stub at path {'.'.join(path_keys_to_delete)}. Configuration merge might be problematic.")
            # Depending on strictness, one might choose to raise an error here.

    cfg.merge_from_dict(final_trial_params_for_merge) 
    _substitute_shared_params_in_config_obj(cfg, current_trial_hyperparameters)

    trial_output_dir = session.get_trial_dir() 
    cfg.work_dir = str(trial_output_dir)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)

    world_size = train.get_context().get_world_size()
    if world_size > 1:
        cfg.launcher = 'pytorch'
    else:
        cfg.launcher = 'none'

    runner = Runner.from_cfg(cfg)
    runner.train()

    metrics_to_report = {}
    if train.get_context().get_world_rank() == 0:
        all_scalar_logs = runner.message_hub.log_scalars 
        
        current_scalar_values = {}
        for key, scalar_log_data in all_scalar_logs.items():
            if scalar_log_data is not None and scalar_log_data.data: 
                current_value = scalar_log_data.current() 
                if current_value is not None:
                    current_scalar_values[key] = float(current_value) 
        
        if current_scalar_values:
            metrics_to_report = current_scalar_values
        else:
            metrics_to_report = {} 
        
        dump_cfg_path = Path(cfg.work_dir) / "final_merged_config.py"
        with open(dump_cfg_path, "w") as f:
            f.write(cfg.pretty_text)
    
    session.report(metrics_to_report)
