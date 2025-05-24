import pprint
from pathlib import Path
import os
from typing import Any, Dict, List, Union, Optional, Tuple 
import copy
import torch 
import time 

from mmengine.config import Config, ConfigDict 
from mmengine.runner import Runner
from ray import train
from ray.air import session
from hod.training.prototypes import perform_prototype_pretraining 
from mmengine.dist import get_dist_info, barrier 

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
    Recursively processes HPO choices.
    (This function should be the one you had previously, responsible for resolving 'Choice' types)
    """
    # Case 1: Current node from trial_config is a list
    if isinstance(current_trial_config_node, list):
        new_list = []
        for i, trial_item in enumerate(current_trial_config_node):
            base_item = None
            if isinstance(current_base_cfg_node, list) and i < len(current_base_cfg_node):
                base_item = current_base_cfg_node[i]
            new_path_for_recursion = current_path_keys + [str(i)]
            new_list.append(_finalize_config_choices(trial_item, base_item, new_path_for_recursion, choice_paths_accumulator))
        return new_list

    # Case 2: Current node from trial_config is not a dictionary (hence a literal)
    if not isinstance(current_trial_config_node, dict):
        return current_trial_config_node

    # Case 3: Current node from trial_config is a dictionary.
    if isinstance(current_base_cfg_node, (dict, ConfigDict)) and \
       current_base_cfg_node.get('type') == 'Choice' and \
       '_CHOICE_TYPE_' in current_trial_config_node:
        
        if current_path_keys: 
            choice_paths_accumulator.append(list(current_path_keys))

        chosen_type_name = current_trial_config_node['_CHOICE_TYPE_']
        
        # If the chosen type itself is Python None, then this node should resolve to actual Python None.
        if chosen_type_name is None:
            return None

        params_for_chosen_type = current_trial_config_node.get(chosen_type_name, {})
        
        final_resolved_node = {'type': chosen_type_name}
        
        original_option_definition_in_base_cfg, option_idx_in_base = _get_original_option_definition(current_base_cfg_node, chosen_type_name)

        if original_option_definition_in_base_cfg is None:
            processed_params = {}
            if isinstance(params_for_chosen_type, dict):
                for p_key, p_value in params_for_chosen_type.items():
                    artificial_path_for_recursion = current_path_keys + [chosen_type_name, p_key]
                    processed_params[p_key] = _finalize_config_choices(
                        p_value, None, artificial_path_for_recursion, choice_paths_accumulator
                    )
            final_resolved_node.update(processed_params)
            return final_resolved_node

        if isinstance(params_for_chosen_type, dict):
            processed_params = {}
            for p_key, p_value in params_for_chosen_type.items():
                original_param_cfg_in_base_option = original_option_definition_in_base_cfg.get(p_key)
                path_to_base_param_definition = current_path_keys + ['options', str(option_idx_in_base), p_key]
                processed_params[p_key] = _finalize_config_choices(
                    p_value, original_param_cfg_in_base_option, path_to_base_param_definition, choice_paths_accumulator 
                )
            final_resolved_node.update(processed_params)
        return final_resolved_node
    else:
        new_dict_for_merging = {}
        for key, trial_value_node in current_trial_config_node.items():
            if isinstance(current_base_cfg_node, (dict, ConfigDict)) and \
               current_base_cfg_node.get('type') == 'Choice': 
                chosen_type_in_trial_node_if_any = current_trial_config_node.get('_CHOICE_TYPE_')
                if key != chosen_type_in_trial_node_if_any and key != '_CHOICE_TYPE_':
                    is_an_option_type_name_in_base_choice = False
                    base_options = current_base_cfg_node.get("options")
                    if isinstance(base_options, list):
                        for opt_stub in base_options:
                            if isinstance(opt_stub, dict) and opt_stub.get("type") == key:
                                is_an_option_type_name_in_base_choice = True
                                break
                    if is_an_option_type_name_in_base_choice:
                        continue
            
            base_cfg_child_node = None
            if isinstance(current_base_cfg_node, (dict, ConfigDict, Config)): 
                base_cfg_child_node = current_base_cfg_node.get(key)

            new_path_for_recursion = current_path_keys + [key]
            new_dict_for_merging[key] = _finalize_config_choices(
                trial_value_node, base_cfg_child_node, new_path_for_recursion, choice_paths_accumulator
            )
        return new_dict_for_merging

def _delete_node_by_path(root_config_node: Union[Config, ConfigDict, dict], path_keys: List[str]) -> bool:
    """
    Deletes a node from a configuration structure.
    (This function should be the one you had previously)
    """
    if not path_keys: return True
    current_parent_for_access = root_config_node
    parent_for_deletion: Union[ConfigDict, dict, None] = None 
    key_to_delete_in_parent = None

    for i, key_segment in enumerate(path_keys):
        is_last_segment = (i == len(path_keys) - 1)
        if isinstance(current_parent_for_access, Config):
            container_to_check = current_parent_for_access._cfg_dict
        elif isinstance(current_parent_for_access, (ConfigDict, dict)):
            container_to_check = current_parent_for_access
        else: return True

        if key_segment not in container_to_check: return True

        if is_last_segment:
            parent_for_deletion = container_to_check
            key_to_delete_in_parent = key_segment
            break
        
        next_node = container_to_check[key_segment]
        if not isinstance(next_node, (Config, ConfigDict, dict)): return True
        current_parent_for_access = next_node

    if parent_for_deletion is not None and key_to_delete_in_parent is not None:
        try:
            del parent_for_deletion[key_to_delete_in_parent]
            return True
        except Exception: return False
    return True

def _substitute_shared_params_in_config_obj(config_node: Any, trial_config: Dict[str, Any]):
    """
    Substitutes 'hpo_ref:' placeholders.
    (This function should be the one you had previously)
    """
    if isinstance(config_node, (Config, ConfigDict)):
        for key in list(config_node.keys()):
            item = config_node[key]
            if isinstance(item, str) and item.startswith("hpo_ref:"):
                ref_key = item.split(":", 1)[1]
                if ref_key in trial_config: config_node[key] = trial_config[ref_key]
            elif isinstance(item, (Config, ConfigDict, dict, list)):
                _substitute_shared_params_in_config_obj(item, trial_config)
    elif isinstance(config_node, dict):
        for key in list(config_node.keys()):
            item = config_node[key]
            if isinstance(item, str) and item.startswith("hpo_ref:"):
                ref_key = item.split(":", 1)[1]
                if ref_key in trial_config: config_node[key] = trial_config[ref_key]
            elif isinstance(item, (Config, ConfigDict, dict, list)):
                _substitute_shared_params_in_config_obj(item, trial_config)
    elif isinstance(config_node, list):
        for i in range(len(config_node)):
            item = config_node[i]
            if isinstance(item, str) and item.startswith("hpo_ref:"):
                ref_key = item.split(":", 1)[1]
                if ref_key in trial_config: config_node[i] = trial_config[ref_key]
            elif isinstance(item, (Config, ConfigDict, dict, list)):
                _substitute_shared_params_in_config_obj(item, trial_config)

def train_loop_per_worker(current_trial_hyperparameters: dict):
    original_cfg_path = current_trial_hyperparameters.pop("cfg_path") 
    project_root_dir = current_trial_hyperparameters.pop("project_root_dir")

    os.chdir(project_root_dir)

    # Load the BASE MMEngine config for HPO processing structure
    base_cfg_for_hpo_processing = Config.fromfile(original_cfg_path) 
    
    # Create a new Config object for this trial, which will be modified.
    cfg = Config.fromfile(original_cfg_path) 

    choice_paths_resolved_in_cfg: List[List[str]] = []

    # Deepcopy current_trial_hyperparameters before passing to _finalize_config_choices
    # as it might be modified if it contains complex objects not handled by simple HPO types.
    trial_params_for_finalizing = copy.deepcopy(current_trial_hyperparameters)

    final_trial_params_for_merge_raw = _finalize_config_choices(
        trial_params_for_finalizing, 
        base_cfg_for_hpo_processing, # Use the reference base config for structure
        [],                          
        choice_paths_resolved_in_cfg 
    )
    
    if not isinstance(final_trial_params_for_merge_raw, dict):
        print(f"CRITICAL ERROR (Rank {train.get_context().get_world_rank()}): final_trial_params_for_merge is not a dict, it is {type(final_trial_params_for_merge_raw)}. This will likely cause a crash. Params: {final_trial_params_for_merge_raw}")
        final_trial_params_for_merge = {}
    else:
        final_trial_params_for_merge = final_trial_params_for_merge_raw

    for path_keys_to_delete in choice_paths_resolved_in_cfg:
        if not path_keys_to_delete: 
            continue
        if not _delete_node_by_path(cfg, path_keys_to_delete): 
            print(f"ERROR (Rank {train.get_context().get_world_rank()}): Failed to delete original Choice stub at path {'.'.join(path_keys_to_delete)}. Configuration merge might be problematic.")

    cfg.merge_from_dict(final_trial_params_for_merge) 
    _substitute_shared_params_in_config_obj(cfg, current_trial_hyperparameters) # Pass original HPO params for hpo_ref

    trial_output_dir = session.get_trial_dir() 
    if trial_output_dir: # Ensure trial_output_dir is not None
        cfg.work_dir = str(trial_output_dir)
        Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    else:
        # Fallback if session.get_trial_dir() is None (should not happen in normal Ray Tune run)
        cfg.work_dir = str(Path(project_root_dir) / "work_dirs" / f"hpo_trial_{time.strftime('%Y%m%d_%H%M%S')}")
        Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
        print(f"Warning: session.get_trial_dir() was None. Using fallback work_dir: {cfg.work_dir}")

    # --- Conditional pre-training based on loss_embed ---
    # If loss_embed is None or not configured, disable prototype pre-training.
    resolved_loss_embed_cfg = cfg.model.bbox_head.get('loss_embed')
    # A choice of 'None' for loss_embed would make resolved_loss_embed_cfg actually None.
    # An empty dict {} or a dict with 'type': None also indicates no effective loss.
    is_loss_embed_none = False
    if not resolved_loss_embed_cfg: # Catches actual None
        is_loss_embed_none = True
    elif isinstance(resolved_loss_embed_cfg, (dict, ConfigDict)):
        if not resolved_loss_embed_cfg: # Catches empty dict {}
            is_loss_embed_none = True
        elif resolved_loss_embed_cfg.get('type') is None: # Catches {'type': None}
            is_loss_embed_none = True
            
    if is_loss_embed_none:
        current_rank_for_log = train.get_context().get_world_rank() if train.get_context() else 'N/A'
        print(f"Rank {current_rank_for_log}: model.bbox_head.loss_embed is effectively None. "
              "Forcing prototype_pretrain_cfg.enable = False.")
        if cfg.get('prototype_pretrain_cfg') is None:
            cfg.prototype_pretrain_cfg = ConfigDict({'enable': False})
        else:
            cfg.prototype_pretrain_cfg.enable = False
    # --- End of conditional pre-training ---

    # --- Prototype Pre-training Integration for HPO ---
    pretrain_checkpoint_to_load = None
    pretrain_cfg_from_hpo = cfg.get('prototype_pretrain_cfg') 

    rank, world_size = get_dist_info()
    # Ensure work_dir is valid for creating files
    work_dir_path = Path(cfg.work_dir)
    pretrain_done_flag_file = work_dir_path / '_pretrain_done.flag'
    pretrain_checkpoint_meta_file = work_dir_path / '_pretrain_checkpoint_path.txt'
    
    actual_pretrain_checkpoint_path_str = "" 

    if pretrain_cfg_from_hpo and pretrain_cfg_from_hpo.get('enable', False):
        if rank == 0:
            print(f"Rank 0: Prototype pre-training enabled via HPO configuration for trial {session.get_trial_id() if session.get_trial_id() else 'N/A'}.")
            
            epochs = pretrain_cfg_from_hpo.get('epochs', 100)
            device_override = pretrain_cfg_from_hpo.get('device')
            force_pretrain = pretrain_cfg_from_hpo.get('force_pretrain', False)
            
            pretrain_output_dir = work_dir_path / 'prototype_pretrain'
            pretrain_output_file = pretrain_output_dir / 'hpo_pretrained_prototypes.pth'
            pretrain_output_dir.mkdir(parents=True, exist_ok=True)

            if not force_pretrain and pretrain_output_file.exists():
                print(f"Rank 0: Found existing HPO pre-trained prototype checkpoint: {pretrain_output_file}")
                print("Rank 0: Skipping pre-training. Set prototype_pretrain_cfg.force_pretrain=True to override.")
                actual_pretrain_checkpoint_path_str = str(pretrain_output_file)
            else:
                if force_pretrain and pretrain_output_file.exists():
                    print(f"Rank 0: force_pretrain=True. Re-running prototype pre-training and overwriting {pretrain_output_file}")
                elif not pretrain_output_file.exists():
                    print(f"Rank 0: No existing HPO pre-trained checkpoint found at {pretrain_output_file}. Starting pre-training.")
                
                pretrain_device = device_override if device_override else ('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Rank 0: Starting prototype pre-training for {epochs} epochs on device '{pretrain_device}'.")
                print(f"Rank 0: Pre-trained checkpoint will be saved to: {pretrain_output_file}")
                
                temp_pretrain_model_config_file = work_dir_path / "hpo_trial_resolved_cfg_for_pretrain.py"
                # Fallback to original_cfg_path if dump fails, though error in dump will be caught by except block.
                model_config_path_to_use_for_pretrain = original_cfg_path 

                try:
                    # Dump the fully HPO-resolved config (self.cfg) to a temporary file
                    cfg.dump(str(temp_pretrain_model_config_file))
                    print(f"Rank 0: Dumped HPO-resolved config for pre-training to {temp_pretrain_model_config_file}")
                    model_config_path_to_use_for_pretrain = str(temp_pretrain_model_config_file)
                    
                    # perform_prototype_pretraining will now use the HPO-resolved config
                    saved_checkpoint_path = perform_prototype_pretraining(
                        model_config_path=model_config_path_to_use_for_pretrain, 
                        output_checkpoint_file=pretrain_output_file,
                        epochs=epochs,
                        device=pretrain_device
                    )
                    print(f"Rank 0: Prototype pre-training complete. Checkpoint saved to: {saved_checkpoint_path}")
                    actual_pretrain_checkpoint_path_str = str(saved_checkpoint_path)
                except Exception as e:
                    print(f"Rank 0: Error during prototype pre-training (or config dump for it): {e}")
                    import traceback
                    traceback.print_exc()                    
                    actual_pretrain_checkpoint_path_str = "" # Ensure this is set on failure
                finally:
                    # Clean up the temporary config file if it was created
                    if temp_pretrain_model_config_file.exists():
                        try:
                            temp_pretrain_model_config_file.unlink()
                            print(f"Rank 0: Cleaned up temporary pre-train config: {temp_pretrain_model_config_file}")
                        except OSError as e_unlink:
                            print(f"Rank 0: Warning - could not delete temporary pre-train config file {temp_pretrain_model_config_file}: {e_unlink}")
            
            pretrain_checkpoint_to_load = actual_pretrain_checkpoint_path_str
        # else rank > 0: wait for rank 0 at the barrier

    else: # Pre-training NOT enabled by HPO config
        if rank == 0:
            print(f"Rank 0: Prototype pre-training is NOT enabled for trial {session.get_trial_id() if session.get_trial_id() else 'N/A'}.")
            actual_pretrain_checkpoint_path_str = "" 
            pretrain_checkpoint_to_load = "" 

    # Rank 0 writes its status and touches the done flag.
    if rank == 0:
        try:
            with open(pretrain_checkpoint_meta_file, 'w') as f:
                f.write(actual_pretrain_checkpoint_path_str)
            pretrain_done_flag_file.touch() 
            print(f"Rank 0: Signaled pre-training status (checkpoint: '{actual_pretrain_checkpoint_path_str}'). Flag: {pretrain_done_flag_file}")
        except Exception as e:
            print(f"Rank 0: CRITICAL ERROR writing pre-train status files: {e}. Other ranks may hang.")

    if world_size > 1:
        print(f"Rank {rank}: Reaching pre-training barrier.")
        barrier() # Synchronize all workers
        print(f"Rank {rank}: Passed pre-training barrier.")

    # Ranks > 0 (and rank 0 if it needs to re-verify) determine the checkpoint path.
    if rank > 0: 
        print(f"Rank {rank}: Attempting to read pre-training status from rank 0. Flag file: {pretrain_done_flag_file}")
        # Add a small timeout loop for the flag file, as barrier might not guarantee file system visibility immediately.
        flag_wait_timeout = 60  # seconds
        flag_wait_start_time = time.time()
        while not pretrain_done_flag_file.exists():
            time.sleep(1)
            if time.time() - flag_wait_start_time > flag_wait_timeout:
                print(f"Rank {rank}: Timeout waiting for pre-train done flag file: {pretrain_done_flag_file}. Proceeding cautiously.")
                break
        
        if pretrain_done_flag_file.exists():
            try:
                with open(pretrain_checkpoint_meta_file, 'r') as f:
                    shared_checkpoint_path_str = f.read().strip()
                if shared_checkpoint_path_str:
                    pretrain_checkpoint_to_load = shared_checkpoint_path_str
                    print(f"Rank {rank}: Received pre-trained checkpoint path: {pretrain_checkpoint_to_load}")
                else:
                    print(f"Rank {rank}: Rank 0 indicated no pre-trained checkpoint to load (skipped, failed, or disabled).")
            except Exception as e:
                print(f"Rank {rank}: Error reading pre-train checkpoint meta file ({pretrain_checkpoint_meta_file}): {e}. Proceeding without.")
        else: 
            print(f"Rank {rank}: Pre-train done flag ('{pretrain_done_flag_file}') not found after barrier and wait. "
                  "Proceeding without pre-trained checkpoint.")

    if pretrain_checkpoint_to_load:
        print(f"Rank {rank}: Setting cfg.load_from = {pretrain_checkpoint_to_load} from pre-training.")
        cfg.load_from = pretrain_checkpoint_to_load
        cfg.resume = False 
    else:
        print(f"Rank {rank}: No pre-trained prototype checkpoint to load for main training.")
    # --- End of Prototype Pre-training Integration ---

    # --- Conditional freezing of embeddings ---
    # If no pre-trained checkpoint was loaded (either disabled, skipped, or failed pre-training),
    # then we must not freeze the embeddings, as they wouldn't be the pre-trained ones.
    if not pretrain_checkpoint_to_load:
        if cfg.model.bbox_head.get('freeze_cls_embeddings', False):
            current_rank_for_log = train.get_context().get_world_rank() if train.get_context() else 'N/A'
            print(f"Rank {current_rank_for_log}: No pre-trained prototype checkpoint loaded. "
                  "Forcing model.bbox_head.freeze_cls_embeddings = False.")
            cfg.model.bbox_head.freeze_cls_embeddings = False
    # --- End of conditional freezing ---

    if world_size > 1:
        cfg.launcher = 'pytorch'
    else:
        cfg.launcher = 'none'
    
    runner = Runner.from_cfg(cfg)
    runner.train()

    metrics_to_report = {}
    if rank == 0: 
        all_scalar_logs = runner.message_hub.log_scalars 
        current_scalar_values = {}
        for key, scalar_log_data in all_scalar_logs.items():
            if scalar_log_data is not None and scalar_log_data.data: 
                current_value = scalar_log_data.current() 
                if current_value is not None:
                    current_scalar_values[key] = float(current_value) 
        
        if current_scalar_values:
            metrics_to_report = current_scalar_values
        
        dump_cfg_path = work_dir_path / "final_merged_config.py"
        with open(dump_cfg_path, "w") as f:
            f.write(cfg.pretty_text)
    
    session.report(metrics_to_report)
