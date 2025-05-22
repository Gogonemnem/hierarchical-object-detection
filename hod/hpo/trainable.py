import pprint
from pathlib import Path
import os
from typing import Any, Dict, List, Union
import copy # Add import for deepcopy
import torch # Add torch import

from mmengine.config import Config, ConfigDict # Ensure ConfigDict is imported
from mmengine.runner import Runner
from ray import train # Changed from 'from ray import tune, train' to just 'from ray import train'
from ray.air import session # Added import for ray.air.session

def _finalize_config_choices(
    current_trial_config_node: Union[Dict[str, Any], List[Any], Any], 
    current_base_cfg_node: Union[Dict[str, Any], List[Any], Any],
    current_path_keys: List[str], # Explicitly pass current path
    choice_paths_accumulator: List[List[str]] # Accumulator for paths of resolved choices
) -> Union[Dict[str, Any], List[Any], Any]:
    """
    Recursively transforms a segment of Ray Tune's trial_config.
    If a 'Choice' is identified (based on '_CHOICE_TYPE_' in trial_config and 'type': 'Choice' in base_cfg),
    it's resolved into a concrete MMEngine component definition and its path is recorded.
    Otherwise, structures are traversed and non-choice parameters are kept.
    """
    if isinstance(current_trial_config_node, list):
        new_list = []
        for i, trial_item in enumerate(current_trial_config_node):
            base_item = None
            if isinstance(current_base_cfg_node, list) and i < len(current_base_cfg_node):
                base_item = current_base_cfg_node[i]
            new_path_for_recursion = current_path_keys + [str(i)]
            new_list.append(_finalize_config_choices(trial_item, base_item, new_path_for_recursion, choice_paths_accumulator))
        return new_list

    if not isinstance(current_trial_config_node, dict):
        return current_trial_config_node

    # DEBUGGING PRINTS:
    # current_path_for_debug = current_path_keys # Use current_path_keys directly
    
    if "loss_embed" in "".join(current_path_keys): # Only print for paths containing loss_embed
        print(f"DEBUG _finalize_config_choices PATH: {'.'.join(current_path_keys)}")
        print(f"  trial_node keys: {list(current_trial_config_node.keys()) if isinstance(current_trial_config_node, dict) else 'Not a dict'}")
        print(f"  base_node is dict: {isinstance(current_base_cfg_node, dict)}")
        if isinstance(current_base_cfg_node, dict):
            print(f"  base_node actual type: {type(current_base_cfg_node)}")
            print(f"  base_node.get('type'): {current_base_cfg_node.get('type')}")
        else:
            print(f"  base_node actual type: {type(current_base_cfg_node)}")
            if current_base_cfg_node is not None:
                if hasattr(current_base_cfg_node, 'type'):
                    try:
                        print(f"  base_node has .type attribute: {current_base_cfg_node.type}")
                    except Exception as e:
                        print(f"  Error accessing base_node.type: {e}")
                if hasattr(current_base_cfg_node, '__getitem__'):
                    try:
                        print(f"  base_node['type'] (if accessible): {current_base_cfg_node['type']}")
                    except Exception as e:
                        print(f"  base_node['type'] not accessible or error: {e}")
                else:
                    print(f"  base_node does not support __getitem__")
            else:
                print(f"  base_node is None")

        if isinstance(current_trial_config_node, dict) and '_CHOICE_TYPE_' in current_trial_config_node:
            print(f"  trial_node has _CHOICE_TYPE_: {current_trial_config_node['_CHOICE_TYPE_']}")
        else:
            print(f"  trial_node does NOT have _CHOICE_TYPE_'")

    if isinstance(current_base_cfg_node, (dict, ConfigDict)) and \
       current_base_cfg_node.get('type') == 'Choice' and \
       '_CHOICE_TYPE_' in current_trial_config_node:
        
        if "loss_embed" in "".join(current_path_keys): # Debug print
            print(f"  CONDITION MET for Choice resolution at path {'.'.join(current_path_keys)}")
        
        # Accumulate the path to this choice node in the base config
        if current_path_keys: 
            # Add a copy of the path to avoid modification issues if lists are reused (though here it's fine)
            # Check if this exact path is already added to prevent duplicates if structure is odd
            # However, each choice stub should be unique by path.
            choice_paths_accumulator.append(list(current_path_keys))

        chosen_type_name = current_trial_config_node['_CHOICE_TYPE_']
        params_for_chosen_type = current_trial_config_node.get(chosen_type_name, {})
        final_resolved_node = {'type': chosen_type_name}
        
        if isinstance(params_for_chosen_type, dict):
            original_option_definition_in_base_cfg = {}
            if isinstance(current_base_cfg_node.get("options"), list):
                for opt_stub in current_base_cfg_node["options"]:
                    if isinstance(opt_stub, dict) and opt_stub.get("type") == chosen_type_name:
                        original_option_definition_in_base_cfg = opt_stub
                        break
            
            processed_params = {}
            for p_key, p_value in params_for_chosen_type.items():
                original_param_cfg_in_base_option = original_option_definition_in_base_cfg.get(p_key)
                # Path for parameter recursion is relative to the trial_config structure being processed
                # This path is for debug/context within recursive calls, not directly for choice_paths_accumulator here.
                # If p_value leads to a nested choice, that recursive call will use its own current_path_keys.
                param_path_for_recursion = current_path_keys + [chosen_type_name, p_key]
                processed_params[p_key] = _finalize_config_choices(
                    p_value, 
                    original_param_cfg_in_base_option,
                    param_path_for_recursion, 
                    choice_paths_accumulator 
                )
            final_resolved_node.update(processed_params)
            
        return final_resolved_node
    else:
        if "loss_embed" in "".join(current_path_keys): # Debug print
            print(f"  CONDITION NOT MET for Choice resolution at path {'.'.join(current_path_keys)}, proceeding to recursive else block.")
        new_dict_for_merging = {}
        for key, trial_value_node in current_trial_config_node.items():
            if isinstance(current_base_cfg_node, (dict, ConfigDict)) and \
               current_base_cfg_node.get('type') == 'Choice' and \
               key != current_trial_config_node.get('_CHOICE_TYPE_') and \
               key != '_CHOICE_TYPE_':
                is_an_option_type_name = False
                if isinstance(current_base_cfg_node.get("options"), list):
                    for opt_stub in current_base_cfg_node["options"]:
                        if isinstance(opt_stub, dict) and opt_stub.get("type") == key:
                            is_an_option_type_name = True
                            break
                if is_an_option_type_name:
                    continue 
            
            base_cfg_child_node = None

            # ---- START NEW DEBUG BLOCK (adapted) ----
            current_path_str_so_far = '.'.join(current_path_keys) # Use current_path_keys
            target_parent_path_str = "model.bbox_head"
            target_child_key = "loss_embed"

            if current_path_str_so_far == target_parent_path_str and key == target_child_key:
                print(f"DEBUG _finalize_config_choices: Processing key='{key}' within parent path='{current_path_str_so_far}'")
                print(f"  Parent node (current_base_cfg_node) type: {type(current_base_cfg_node)}")
                if isinstance(current_base_cfg_node, (dict, ConfigDict)):
                    print(f"  Parent node keys: {list(current_base_cfg_node.keys())}")
                    if target_child_key in current_base_cfg_node:
                        child_via_bracket = current_base_cfg_node[target_child_key]
                        print(f"  Parent_node['{target_child_key}'] exists. Type: {type(child_via_bracket)}")
                        if isinstance(child_via_bracket, (dict, ConfigDict)):
                            print(f"    Parent_node['{target_child_key}'].get('type'): {child_via_bracket.get('type')}")
                    else:
                        print(f"  Parent_node does NOT contain key '{target_child_key}' via __contains__.")
                    
                    child_via_get = current_base_cfg_node.get(target_child_key)
                    print(f"  Parent_node.get('{target_child_key}') returns type: {type(child_via_get)}")
                    if child_via_get is None:
                        print(f"  WARNING: Parent_node.get('{target_child_key}') returned None.")
                    else:
                        if isinstance(child_via_get, (dict, ConfigDict)):
                             print(f"    Parent_node.get('{target_child_key}').get('type'): {child_via_get.get('type')}")
                else:
                    print(f"  Parent node (current_base_cfg_node) is NOT a dict/ConfigDict instance.")
            # ---- END NEW DEBUG BLOCK ----
            
            if isinstance(current_base_cfg_node, (dict, ConfigDict, Config)): 
                base_cfg_child_node = current_base_cfg_node.get(key)
                # ---- START DIAGNOSTIC PRINT (adapted) ----
                diag_current_parent_path_list = current_path_keys # Use current_path_keys
                diag_current_parent_path_str = '.'.join(diag_current_parent_path_list)
                diag_expected_parent_path = "model.bbox_head"
                diag_target_child_key_for_print = "loss_embed"

                if diag_current_parent_path_str == diag_expected_parent_path and key == diag_target_child_key_for_print:
                    print(f"DIAGNOSTIC @{diag_current_parent_path_str} (processing child key '{key}'):")
                    print(f"  Assigned base_cfg_child_node. Type: {type(base_cfg_child_node)}")
                    if base_cfg_child_node is None:
                        print(f"  WARNING: base_cfg_child_node IS NONE immediately after parent.get('{key}').")
                    elif isinstance(base_cfg_child_node, (dict, ConfigDict)):
                        print(f"  base_cfg_child_node is ConfigDict/dict. Has .get('type'): {base_cfg_child_node.get('type')}")
                    else:
                        print(f"  base_cfg_child_node is not None but not dict/ConfigDict. Actual type: {type(base_cfg_child_node)}")
                # ---- END DIAGNOSTIC PRINT ----
            
            # ---- START PRE-RECURSION PRINT (adapted) ----
            pre_recursion_path_list = current_path_keys # Use current_path_keys
            pre_recursion_path_str = '.'.join(pre_recursion_path_list)

            if (pre_recursion_path_str == "" and key == "model") or \
               (pre_recursion_path_str == "model" and key == "bbox_head") or \
               (pre_recursion_path_str == "model.bbox_head" and key == "loss_embed"):
                print(f"PRE-RECURSION @parent_path='{pre_recursion_path_str}', for child_key='{key}':")
                print(f"  Value of base_cfg_child_node (to be passed as next current_base_cfg_node): {type(base_cfg_child_node)}")
                if isinstance(base_cfg_child_node, (dict, ConfigDict)):
                    print(f"    It's a dict/ConfigDict. Keys: {list(base_cfg_child_node.keys())}")
                    print(f"    Its .get('type') attribute: {base_cfg_child_node.get('type')}")
                elif base_cfg_child_node is None:
                    print(f"    WARNING: base_cfg_child_node is None.")
                    print(f"      Parent ({pre_recursion_path_str}) type was: {type(current_base_cfg_node)}")
                    if isinstance(current_base_cfg_node, (dict, ConfigDict, Config)):
                         print(f"      Parent keys: {list(current_base_cfg_node.keys())}")
                         if key not in current_base_cfg_node:
                             print(f"      Parent does NOT contain key '{key}'.")
                         else:
                             print(f"      Parent.get('{key}') explicitly returned None (or parent['{key}'] is None).")
                    elif current_base_cfg_node is None:
                         print(f"      Parent itself was None when trying to get child '{key}'.")
            # ---- END PRE-RECURSION PRINT ----

            new_path_for_recursion = current_path_keys + [key]
            new_dict_for_merging[key] = _finalize_config_choices(
                trial_value_node, 
                base_cfg_child_node, 
                new_path_for_recursion, 
                choice_paths_accumulator
            )
        return new_dict_for_merging

def _substitute_shared_params_in_config_obj(config_node: Any, trial_config: Dict[str, Any]):
    """
    Recursively traverses the config structure (dict, list, Config, ConfigDict)
    and replaces string values starting with "hpo_ref:" with the corresponding
    values from trial_config.
    """
    if isinstance(config_node, (Config, ConfigDict)): # MMEngine Config or ConfigDict
        for key in list(config_node.keys()): # Iterate over a copy of keys for safe modification
            item = config_node[key]
            if isinstance(item, str) and item.startswith("hpo_ref:"):
                ref_key = item.split(":", 1)[1]
                if ref_key in trial_config:
                    config_node[key] = trial_config[ref_key]
                    print(f"DEBUG SHARED HPO: Substituted '{item}' with '{trial_config[ref_key]}' at path ending with '{key}'")
                else:
                    print(f"Warning: HPO reference '{ref_key}' not found in trial_config for config key '{key}'. Original value '{item}' kept.")
            elif isinstance(item, (Config, ConfigDict, dict, list)):
                _substitute_shared_params_in_config_obj(item, trial_config)
    elif isinstance(config_node, dict): # Standard Python dict
        for key in list(config_node.keys()):
            item = config_node[key]
            if isinstance(item, str) and item.startswith("hpo_ref:"):
                ref_key = item.split(":", 1)[1]
                if ref_key in trial_config:
                    config_node[key] = trial_config[ref_key]
                    print(f"DEBUG SHARED HPO: Substituted '{item}' with '{trial_config[ref_key]}' at path ending with '{key}'")
                else:
                    print(f"Warning: HPO reference '{ref_key}' not found in trial_config for dict key '{key}'. Original value '{item}' kept.")
            elif isinstance(item, (Config, ConfigDict, dict, list)):
                _substitute_shared_params_in_config_obj(item, trial_config)
    elif isinstance(config_node, list): # Standard Python list
        for i in range(len(config_node)):
            item = config_node[i]
            if isinstance(item, str) and item.startswith("hpo_ref:"):
                ref_key = item.split(":", 1)[1]
                if ref_key in trial_config:
                    config_node[i] = trial_config[ref_key]
                    print(f"DEBUG SHARED HPO: Substituted '{item}' with '{trial_config[ref_key]}' at list index {i}")
                else:
                    print(f"Warning: HPO reference '{ref_key}' not found in trial_config for list item at index {i}. Original value '{item}' kept.")
            elif isinstance(item, (Config, ConfigDict, dict, list)):
                _substitute_shared_params_in_config_obj(item, trial_config)
    # Non-container types or non-hpo_ref strings are left as is by this function.

def train_loop_per_worker(current_trial_hyperparameters: dict):
    """
    This function is executed by each DDP worker process managed by TorchTrainer.
    It receives the hyperparameter configuration for the current trial.
    Static configuration like cfg_path and project_root_dir are expected to be in current_trial_hyperparameters.
    """
    # Extract static and dynamic config parts
    # Static parts (like original cfg_path, project_root_dir) are passed via train_loop_config
    # and merged into current_trial_hyperparameters by Ray Tune.
    cfg_path = current_trial_hyperparameters.pop("cfg_path")
    project_root_dir = current_trial_hyperparameters.pop("project_root_dir")
    # The remaining items in current_trial_hyperparameters are the actual hyperparameters for this trial.

    print(f"DEBUG: train_loop_per_worker (Rank {train.get_context().get_world_rank()}): Initial CWD: {os.getcwd()}")
    print(f"DEBUG: train_loop_per_worker (Rank {train.get_context().get_world_rank()}): Received cfg_path: {cfg_path}")
    print(f"DEBUG: train_loop_per_worker (Rank {train.get_context().get_world_rank()}): Received project_root_dir: {project_root_dir}")

    print(f"--- train_loop_per_worker (Rank {train.get_context().get_world_rank()}) received hyperparameters: ---")
    pprint.pprint(current_trial_hyperparameters)
    print(f"--- Base Cfg Path: {cfg_path} ---")
    print(f"--- Project Root: {project_root_dir} ---")
    print("--------------------------------------------------")

    os.chdir(project_root_dir)
    print(f"Changed CWD to: {os.getcwd()} (Rank {train.get_context().get_world_rank()})")

    cfg = Config.fromfile(cfg_path)
    
    choice_paths_resolved_in_cfg: List[List[str]] = []

    # current_trial_hyperparameters now only contains the HPO choices
    final_trial_params_for_merge = _finalize_config_choices(
        copy.deepcopy(current_trial_hyperparameters), 
        cfg,                         
        [],                          
        choice_paths_resolved_in_cfg 
    )
    
    # ... (Pre-merge deletion logic for choice stubs - remains the same) ...
    # Before merging, delete the original choice stubs from cfg
    for path_keys_to_delete in choice_paths_resolved_in_cfg:
        if not path_keys_to_delete:
            continue
        
        current_node = cfg
        parent_node = None
        key_of_node_to_delete = None

        # Navigate to the parent of the node to be deleted
        for i, key_segment in enumerate(path_keys_to_delete):
            if isinstance(current_node, (Config, ConfigDict, dict)):
                if key_segment in current_node:
                    if i == len(path_keys_to_delete) - 1: # This is the key for the node to delete
                        parent_node = current_node
                        key_of_node_to_delete = key_segment
                        break 
                    current_node = current_node[key_segment]
                else:
                    print(f"Warning (Rank {train.get_context().get_world_rank()}): Path segment '{key_segment}' not found in cfg during pre-merge deletion for path {'.'.join(path_keys_to_delete)}")
                    parent_node = None # Path broken
                    break
            else:
                print(f"Warning (Rank {train.get_context().get_world_rank()}): Node at '{'.'.join(path_keys_to_delete[:i])}' is not a dict-like object during pre-merge deletion for path {'.'.join(path_keys_to_delete)}")
                parent_node = None # Path broken
                break
        
        if parent_node is not None and key_of_node_to_delete is not None:
            try:
                print(f"Pre-merge (Rank {train.get_context().get_world_rank()}): Deleting cfg node at path: {'.'.join(path_keys_to_delete)}")
                del parent_node[key_of_node_to_delete]
            except Exception as e:
                print(f"Error deleting cfg node at path {'.'.join(path_keys_to_delete)} (Rank {train.get_context().get_world_rank()}): {e}")
        else:
            print(f"Warning (Rank {train.get_context().get_world_rank()}): Could not delete cfg node at path {'.'.join(path_keys_to_delete)} before merge. Navigation failed or key not found at final step.")

    cfg.merge_from_dict(final_trial_params_for_merge)
    _substitute_shared_params_in_config_obj(cfg, current_trial_hyperparameters) # Pass HPO params for substitution refs

    # Set up work_dir using Ray AIR session
    trial_output_dir = session.get_trial_dir() # Changed to use session.get_trial_dir()
    cfg.work_dir = str(trial_output_dir)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    print(f"DEBUG (Rank {train.get_context().get_world_rank()}): work_dir set to: {cfg.work_dir}")

    # Launcher setup for MMEngine
    # When using TorchTrainer with num_workers > 1, Ray Train sets up DDP.
    # MMEngine should use 'pytorch' launcher.
    # If num_workers == 1 (single GPU trial), 'none' launcher.
    world_size = train.get_context().get_world_size()
    if world_size > 1:
        cfg.launcher = 'pytorch'
        print(f"DEBUG (Rank {train.get_context().get_world_rank()}): world_size={world_size}. Setting cfg.launcher = 'pytorch'.")
    else:
        cfg.launcher = 'none'
        print(f"DEBUG (Rank {train.get_context().get_world_rank()}): world_size={world_size}. Setting cfg.launcher = 'none'.")

    # DDP environment variables are automatically set by Ray Train / TorchTrainer.
    # No need to print them here unless for deep debugging, MMEngine will pick them up.

    # The user's debug exception can be removed or conditionalized for rank 0 if needed.
    # if train.get_context().get_world_rank() == 0:
    #     raise Exception("DEBUG: Stopping execution here to inspect environment variables before Runner.from_cfg.")

    runner = Runner.from_cfg(cfg)
    runner.train()

    # Metric reporting - ALL workers must call session.report()
    # Rank 0 reports the actual metrics for Ray Tune.
    # Other ranks report a dummy/empty dict to satisfy the trainer.
    metrics_to_report = {}
    if train.get_context().get_world_rank() == 0:
        map_metric_keys = ['val/coco/bbox_mAP', 'coco/bbox_mAP', 'val/mAP', 'mAP']
        best_map = None
        for key in map_metric_keys:
            scalar_values = runner.message_hub.get_scalar(key)
            if scalar_values is not None and scalar_values.data:
                best_map = scalar_values.current()
                if best_map is not None:
                    print(f"Rank 0: Found metric {key} with value: {best_map}")
                    break
        
        if best_map is None:
            mf1_values = runner.message_hub.get_scalar("coco/bbox_mF1")
            if mf1_values is not None and mf1_values.data:
                best_map = mf1_values.current()
                print(f"Rank 0: Found metric coco/bbox_mF1 with value: {best_map}")

        if best_map is not None:
            metrics_to_report = {"mAP": best_map}
        else:
            print("Rank 0: Warning: Could not find a suitable mAP or mF1 metric to report to Ray Tune.")
            metrics_to_report = {"mAP": 0.0}
        
        # Save final config only on Rank 0
        dump_cfg_path = Path(cfg.work_dir) / "final_merged_config.py"
        with open(dump_cfg_path, "w") as f:
            f.write(cfg.pretty_text)
        print(f"Rank 0: Final merged config saved to {dump_cfg_path}")
    
    # All workers call session.report()
    # Rank 0 sends actual metrics, others can send an empty dict or minimal status.
    # For simplicity, non-rank 0 workers will also report the metrics_to_report dict,
    # which will be empty for them. Ray Tune primarily uses Rank 0's report for optimization.
    session.report(metrics_to_report)

# The old train_trial function is no longer directly used by Ray Tune with TorchTrainer.
# It can be removed or kept for other purposes if needed, but for this HPO setup,
# train_loop_per_worker is the key function.
# For clarity, I will comment out the old train_trial function.

# def train_trial(trial_config: dict, cfg_path: str, project_root_dir: str):
#     # ... (old implementation)
#     pass
