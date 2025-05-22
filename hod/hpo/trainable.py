import pprint
from pathlib import Path
import os
from typing import Any, Dict, List, Union
import copy # Add import for deepcopy

from mmengine.config import Config, ConfigDict # Ensure ConfigDict is imported
from mmengine.runner import Runner
from ray import tune, train

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

def train_trial(trial_config: dict, cfg_path: str, project_root_dir: str):
    # ... (initial prints and os.chdir remain the same) ...
    print(f"DEBUG: train_trial: Initial CWD: {os.getcwd()}")
    print(f"DEBUG: train_trial: Received cfg_path: {cfg_path}")
    print(f"DEBUG: train_trial: Received project_root_dir: {project_root_dir}")

    print("--- train_trial received config from Ray Tune: ---")
    pprint.pprint(trial_config)
    print(f"--- Base Cfg Path: {cfg_path} ---")
    print(f"--- Project Root: {project_root_dir} ---")
    print("--------------------------------------------------")

    os.chdir(project_root_dir)
    print(f"Changed CWD to: {os.getcwd()}")

    cfg = Config.fromfile(cfg_path)
    
    print("--- train_trial: Base cfg BEFORE _finalize_config_choices AND pre-merge deletion ---")
    # ... (existing inspection of base cfg can remain) ...
    target_node_path_str = "model.bbox_head.loss_embed" 
    try:
        node_to_inspect = cfg['model']['bbox_head']['loss_embed']
        print(f"--- Inspecting BASE cfg.{target_node_path_str} specifically: ---")
        pprint.pprint(node_to_inspect) 
        print(f"Type of BASE cfg.{target_node_path_str}: {type(node_to_inspect)}")
        is_dict_check = isinstance(node_to_inspect, dict)
        print(f"BASE cfg.{target_node_path_str} isinstance(node, dict): {is_dict_check}")
        if is_dict_check:
            node_type_attr = node_to_inspect.get('type')
            print(f"BASE cfg.{target_node_path_str}.get('type'): {node_type_attr}")
        else:
            if hasattr(node_to_inspect, 'type'):
                 print(f"BASE cfg.{target_node_path_str} is not a dict, but has 'type' attr: {getattr(node_to_inspect, 'type')}")
            else:
                 print(f"BASE cfg.{target_node_path_str} is not a dict and does not have .get('type') or a 'type' attribute.")
    except KeyError:
        print(f"BASE cfg.{target_node_path_str} not found via key access in base_cfg.")
    except Exception as e:
        print(f"Error inspecting base_cfg.{target_node_path_str}: {type(e).__name__} - {e}")
    print("-------------------------------------------------------------")

    choice_paths_resolved_in_cfg: List[List[str]] = [] 

    final_trial_params_for_merge = _finalize_config_choices(
        copy.deepcopy(trial_config), 
        cfg,                         
        [],                          # Initial path_keys is empty
        choice_paths_resolved_in_cfg # Pass the accumulator
    )
    
    print("--- train_trial: Final trial params for merge (after _finalize_config_choices): ---")
    pprint.pprint(final_trial_params_for_merge)
    print("------------------------------------------------------------------------------------")

    print("--- train_trial: Paths of choices resolved in base_cfg (to be deleted before merge): ---")
    pprint.pprint(choice_paths_resolved_in_cfg)
    print("---------------------------------------------------------------------------------------")

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
                    print(f"Warning: Path segment '{key_segment}' not found in cfg during pre-merge deletion for path {'.'.join(path_keys_to_delete)}")
                    parent_node = None # Path broken
                    break
            else:
                print(f"Warning: Node at '{'.'.join(path_keys_to_delete[:i])}' is not a dict-like object during pre-merge deletion for path {'.'.join(path_keys_to_delete)}")
                parent_node = None # Path broken
                break
        
        if parent_node is not None and key_of_node_to_delete is not None:
            try:
                print(f"Pre-merge: Deleting cfg node at path: {'.'.join(path_keys_to_delete)}")
                del parent_node[key_of_node_to_delete]
            except Exception as e:
                print(f"Error deleting cfg node at path {'.'.join(path_keys_to_delete)}: {e}")
        else:
            print(f"Warning: Could not delete cfg node at path {'.'.join(path_keys_to_delete)} before merge. Navigation failed or key not found at final step.")

    cfg.merge_from_dict(final_trial_params_for_merge)

    print("--- train_trial: cfg object AFTER merge, BEFORE shared param substitution ---")
    # You can add a pprint of cfg here if needed for debugging
    
    # Perform shared parameter substitution
    _substitute_shared_params_in_config_obj(cfg, trial_config)
    
    print("--- train_trial: cfg object AFTER shared param substitution, BEFORE Runner.from_cfg ---")
    # You can add a pprint of cfg here if needed for debugging

    print("--- train_trial: Inspecting cfg.model.bbox_head.loss_cls after merge AND substitution ---") 
    if hasattr(cfg, 'model') and cfg.model is not None and \
       hasattr(cfg.model, 'bbox_head') and cfg.model.bbox_head is not None and \
       hasattr(cfg.model.bbox_head, 'loss_cls') and cfg.model.bbox_head.loss_cls is not None:
        pprint.pprint(cfg.model.bbox_head.loss_cls)
    else:
        print("cfg.model.bbox_head.loss_cls not found or structure is different.")
    print("------------------------------------------------------------------------------------")

    current_trial_dir = train.get_context().get_trial_dir()
    if not current_trial_dir:
        print("ERROR: Could not get trial directory from Ray Tune context!")
        current_trial_dir = Path(project_root_dir) / "work_dirs" / "ray_trials" / f"trial_{train.get_context().get_trial_id()}"
        print(f"Falling back to trial directory: {current_trial_dir}")
    
    trial_dir = Path(current_trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir = str(trial_dir)

    runner = Runner.from_cfg(cfg)
    runner.train()

    # Assuming 'coco/bbox_mAP' is the metric. Adjust if it's different (e.g., 'val/mAP')
    # Check common MMDetection metric names if this is not found.
    # Common names: 'val/coco/bbox_mAP', 'test/coco/bbox_mAP', or just 'mAP' if configured.
    # Based on previous logs, 'coco/bbox_mF1' was used, let's stick to that or a more general mAP.
    # Try to get a common mAP metric first.
    map_metric_keys = ['val/coco/bbox_mAP', 'coco/bbox_mAP', 'val/mAP', 'mAP']
    best_map = None
    for key in map_metric_keys:
        scalar_values = runner.message_hub.get_scalar(key) # get_scalar returns a MessageHubStore object
        if scalar_values is not None and scalar_values.data: # Check if data exists
             # Get the last value if it's a series, or the value itself
            best_map = scalar_values.current() # .current() gives the latest value
            if best_map is not None:
                print(f"Found metric {key} with value: {best_map}")
                break
    
    if best_map is None: # Fallback to previously used mF1 if mAP not found
        mf1_values = runner.message_hub.get_scalar("coco/bbox_mF1")
        if mf1_values is not None and mf1_values.data:
            best_map = mf1_values.current()
            print(f"Found metric coco/bbox_mF1 with value: {best_map}")

    if best_map is not None:
        train.report({"mAP": best_map}) # Report the found metric as mAP
    else:
        print("Warning: Could not find a suitable mAP or mF1 metric to report to Ray Tune.")
        train.report({"mAP": 0.0}) # Report a default value if nothing found

    dump_cfg_path = trial_dir / "final_merged_config.py"
    print(f"DEBUG: trainable.py: Final merged config object before pretty_text: {type(cfg)}")
    # print(f"DEBUG: trainable.py: cfg.pretty_text type: {type(cfg.pretty_text)}") # This would execute pretty_text
    # print(f"DEBUG: trainable.py: cfg.filename value: {cfg.filename}")
    # print(f"DEBUG: trainable.py: cfg.text type: {type(cfg.text)}")

    with open(dump_cfg_path, "w") as f:
        f.write(cfg.pretty_text) # Changed from cfg.pretty_text()

    # Clean up the temporary config file if it was created
