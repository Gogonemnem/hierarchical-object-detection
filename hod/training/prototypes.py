import torch
from mmengine.config import Config
from mmdet.registry import MODELS
from mmengine.registry import OPTIM_WRAPPERS, OPTIMIZERS
from mmdet.utils import register_all_modules
from pathlib import Path
import copy

# Import your EmbeddingClassifier, assuming path
from hod.models.layers import EmbeddingClassifier
# from hod.models.losses import HierarchicalContrastiveLoss, EntailmentConeLoss # If needed for isinstance checks on losses

register_all_modules() 

def perform_prototype_pretraining(
    model_config_path: str, 
    output_checkpoint_file: Path, 
    epochs: int, 
    device: str
) -> Path:
    """
    Performs prototype pre-training based on the provided model configuration.

    Args:
        model_config_path: Path to the model config file.
        output_checkpoint_file: Path where the pre-trained checkpoint will be saved.
        epochs: Number of training epochs.
        device: Device to use for training (e.g., 'cuda', 'cpu').

    Returns:
        Path to the saved pre-trained checkpoint.
        
    Raises:
        FileNotFoundError: If the config file is not found.
        ValueError: If critical components (model, classifier, loss, optimizer) cannot be built or found.
        RuntimeError: For other errors during training or setup.
    """
    # Ensure parent directory for output checkpoint exists
    output_checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    # Load configuration
    try:
        cfg = Config.fromfile(model_config_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading config file {model_config_path}: {e}")

    # Build the full model from config
    try:
        model = MODELS.build(cfg.model).to(device)
        model.train() # Set model to training mode
    except Exception as e:
        raise ValueError(f"Error building model from config {model_config_path}: {e}")

    # --- Determine loss_embed_cfg and identify the module associated with it ---
    loss_embed_cfg = None
    associated_module_config_path = "unknown" # For error reporting
    # The actual module instance in the built model that corresponds to this config path
    module_for_primary_classifier = None 

    if hasattr(cfg.model, 'bbox_head'):
        cfg_bbox_head = cfg.model.bbox_head
        # Ensure model_bbox_head is accessed only if cfg_bbox_head exists
        model_bbox_head = model.bbox_head if hasattr(model, 'bbox_head') else None

        if hasattr(cfg_bbox_head, 'cls_branches') and \
           isinstance(cfg_bbox_head.cls_branches, list) and len(cfg_bbox_head.cls_branches) > 0 and \
           'loss_embed' in cfg_bbox_head.cls_branches[0]:
            loss_embed_cfg = cfg_bbox_head.cls_branches[0].loss_embed
            associated_module_config_path = "model.bbox_head.cls_branches[0]"
            if model_bbox_head and hasattr(model_bbox_head, 'cls_branches') and \
               isinstance(model_bbox_head.cls_branches, torch.nn.ModuleList) and \
               len(model_bbox_head.cls_branches) > 0:
                module_for_primary_classifier = model_bbox_head.cls_branches[0]
        elif 'loss_embed' in cfg_bbox_head:
            loss_embed_cfg = cfg_bbox_head.loss_embed
            associated_module_config_path = "model.bbox_head"
            if model_bbox_head:
                module_for_primary_classifier = model_bbox_head
    
    if loss_embed_cfg is None:
        raise ValueError(
            "Could not find loss_embed configuration in expected locations "
            "(e.g., cfg.model.bbox_head.cls_branches[0].loss_embed or cfg.model.bbox_head.loss_embed)."
        )
    
    try:
        loss_fn = MODELS.build(loss_embed_cfg).to(device)
        print(f"Info: Successfully built loss function '{loss_embed_cfg.type}' for pre-training, from {associated_module_config_path}.")
    except Exception as e:
        raise ValueError(f"Error building loss function from config ({associated_module_config_path}): {e}")
    # --- End loss_embed_cfg ---

    # --- Identify primary_embedding_classifier based on module_for_primary_classifier ---
    primary_embedding_classifier = None
    if module_for_primary_classifier:
        if isinstance(module_for_primary_classifier, EmbeddingClassifier):
            primary_embedding_classifier = module_for_primary_classifier
        elif hasattr(module_for_primary_classifier, 'fc_cls') and \
             isinstance(module_for_primary_classifier.fc_cls, EmbeddingClassifier):
            primary_embedding_classifier = module_for_primary_classifier.fc_cls
            
    if not primary_embedding_classifier:
        # Fallback: if the specific module didn't yield an EC, search generically.
        # This might indicate a config/model structure mismatch with where loss_embed_cfg was defined.
        print(f"Warning: Module at {associated_module_config_path} (where loss_embed_cfg was found) "
              "is not/does not directly contain the EmbeddingClassifier. Searching generically for primary EC.")
        for m in model.modules():
            if isinstance(m, EmbeddingClassifier):
                primary_embedding_classifier = m
                print(f"Info: Found a primary EmbeddingClassifier via generic search: {type(primary_embedding_classifier).__name__}. "
                      "Ensure this aligns with the intent for pre-training loss.")
                break
        if not primary_embedding_classifier:
            raise ValueError(
                f"The module ({type(module_for_primary_classifier).__name__} at {associated_module_config_path}) "
                "associated with loss_embed_cfg is not an EmbeddingClassifier nor does it contain a .fc_cls "
                "EmbeddingClassifier. A generic search also failed to find any EmbeddingClassifier."
            )
    
    print(f"Info: Using primary EmbeddingClassifier: {type(primary_embedding_classifier).__name__} for optimization.")
    # --- End primary_embedding_classifier identification ---

    # --- Collect ALL EmbeddingClassifier instances for parameter copying ---
    all_embedding_classifiers_for_copy = []
    for m_ in model.modules():
        if isinstance(m_, EmbeddingClassifier):
            if m_ not in all_embedding_classifiers_for_copy: # Ensure uniqueness
                 all_embedding_classifiers_for_copy.append(m_)
    
    if not all_embedding_classifiers_for_copy:
        # This case should ideally not be hit if primary_embedding_classifier was found
        raise ValueError("No EmbeddingClassifier instances found in the model for copying, though a primary was identified.")
    print(f"Info: Found {len(all_embedding_classifiers_for_copy)} EmbeddingClassifier instance(s) for parameter copying.")
    # --- End Collect ALL ---

    # Determine parameters to optimize
    params_to_optimize = []
    # ... (rest of the function from this point, with 'embedding_classifier' replaced by 'primary_embedding_classifier')
    # Use primary_embedding_classifier for getting .embeddings, .prototypes, .logit_scale
    if hasattr(primary_embedding_classifier, 'embeddings') and isinstance(primary_embedding_classifier.embeddings, torch.nn.Parameter):
        params_to_optimize.append(primary_embedding_classifier.embeddings)
    elif hasattr(primary_embedding_classifier, 'prototypes') and isinstance(primary_embedding_classifier.prototypes, torch.nn.Parameter):
        params_to_optimize.append(primary_embedding_classifier.prototypes)
    else:
        raise ValueError(f"{type(primary_embedding_classifier).__name__}.embeddings (or .prototypes) not found or not a Parameter.")

    loss_type_from_config = loss_embed_cfg.type # This is already available
    if 'HierarchicalContrastiveLoss' in loss_type_from_config:
        if hasattr(primary_embedding_classifier, 'logit_scale') and isinstance(primary_embedding_classifier.logit_scale, torch.nn.Parameter):
            params_to_optimize.append(primary_embedding_classifier.logit_scale)
        else:
            print(f"Warning: HierarchicalContrastiveLoss is used, but logit_scale not found or not a Parameter in primary {type(primary_embedding_classifier).__name__}. It will not be optimized.")
    
    if not params_to_optimize:
        raise ValueError("No parameters to optimize for pre-training from the primary EmbeddingClassifier.")

    # Build OptimWrapper from config
    optim_wrapper_instance = None
    try:
        if not hasattr(cfg, 'optim_wrapper'):
            raise ValueError("Optimizer wrapper configuration (cfg.optim_wrapper) not found in config.")
        
        # Deepcopy the optimizer part of the config to avoid modifying the original
        optimizer_cfg_original = cfg.optim_wrapper.get('optimizer')
        if optimizer_cfg_original is None:
            raise ValueError("cfg.optim_wrapper.optimizer key is missing.")
        if not isinstance(optimizer_cfg_original, (dict, Config)):
            raise ValueError(f"cfg.optim_wrapper.optimizer is not a dictionary-like object, got {type(optimizer_cfg_original)}.")
            
        optimizer_config = copy.deepcopy(optimizer_cfg_original) # Use deepcopy
        optimizer_config['params'] = params_to_optimize # Set the specific parameters to optimize
        
        # Build the optimizer with only the specified parameters
        actual_optimizer = OPTIMIZERS.build(optimizer_config)
        
        # Deepcopy the optim_wrapper base config and set the new optimizer
        optim_wrapper_base_cfg = copy.deepcopy(cfg.optim_wrapper) # Use deepcopy
        optim_wrapper_base_cfg['optimizer'] = actual_optimizer
        
        # If paramwise_cfg exists, it might apply to the selected params if their names match.
        # For pre-training, usually, a simpler setup is desired.
        # If you want to *ignore* main training's paramwise_cfg for pre-training,
        # you could del optim_wrapper_base_cfg['paramwise_cfg'] if it exists.
        # However, if the pre-trainable params (embeddings, logit_scale) have specific rules
        # in paramwise_cfg that you *want* to apply, then leave it.
        # For now, let's assume we want to keep it simple and potentially ignore complex paramwise rules
        # that might not apply or might be problematic for this limited set of parameters.
        if 'paramwise_cfg' in optim_wrapper_base_cfg:
            print("Info: 'paramwise_cfg' found in optim_wrapper. Removing for pre-training OptimWrapper construction.")
            del optim_wrapper_base_cfg['paramwise_cfg'] # Ensure this is active

        optim_wrapper_instance = OPTIM_WRAPPERS.build(optim_wrapper_base_cfg)
        
        effective_lr = optimizer_config.get('lr', 'N/A')
        optimizer_type_str = optimizer_config.get('type', 'N/A')
        print(f"Info: Successfully built OptimWrapper for pre-training with optimizer: {optimizer_type_str} with LR: {effective_lr}")

    except Exception as e:
        raise RuntimeError(f"Error building OptimWrapper from config for pre-training: {e}")

    print(f"Info: Starting pre-training of prototypes for {epochs} epochs on device '{device}'.")
    print(f"Info: Using loss: {loss_type_from_config}")
    print(f"Info: Optimizing {len(params_to_optimize)} parameter group(s) from {type(primary_embedding_classifier).__name__}.")

    for epoch in range(epochs):
        optim_wrapper_instance.optimizer.zero_grad()
        
        current_prototypes_for_loss = None
        if hasattr(primary_embedding_classifier, 'embeddings') and isinstance(primary_embedding_classifier.embeddings, torch.nn.Parameter):
            current_prototypes_for_loss = primary_embedding_classifier.embeddings
        elif hasattr(primary_embedding_classifier, 'prototypes') and isinstance(primary_embedding_classifier.prototypes, torch.nn.Parameter):
            current_prototypes_for_loss = primary_embedding_classifier.prototypes
        else:
            raise ValueError("Could not retrieve embeddings/prototypes from primary classifier for training loop.")

        loss_value = torch.tensor(0.0, device=device)

        if 'HierarchicalContrastiveLoss' in loss_type_from_config:
            if not hasattr(primary_embedding_classifier, 'get_distance_logits'):
                raise ValueError(f"Primary {type(primary_embedding_classifier).__name__} is missing 'get_distance_logits' method, required for HCL pre-training.")
            logits_matrix = primary_embedding_classifier.get_distance_logits(current_prototypes_for_loss.unsqueeze(0), current_prototypes_for_loss)
            loss_value = loss_fn(logits_matrix.squeeze(0))
        elif 'EntailmentConeLoss' in loss_type_from_config:
            loss_value = loss_fn(current_prototypes_for_loss)
        else:
            raise ValueError(f"Loss type {loss_type_from_config} from config is not supported by this pre-training script's loop.")

        loss_value.backward()
        optim_wrapper_instance.step()

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            log_msg = f"Pre-train Epoch [{epoch+1}/{epochs}], Loss: {loss_value.item():.4f}"
            if 'HierarchicalContrastiveLoss' in loss_type_from_config and hasattr(primary_embedding_classifier, 'logit_scale'):
                log_msg += f", LogitScale: {primary_embedding_classifier.logit_scale.item():.4f}"
            print(log_msg)
    
    # --- Copy optimized parameters to all identified EmbeddingClassifiers ---
    print(f"Info: Copying optimized parameters from primary {type(primary_embedding_classifier).__name__} to {len(all_embedding_classifiers_for_copy)} EmbeddingClassifier instance(s).")
    
    optimized_embeddings_data = None
    # Determine which attribute holds the embeddings in the primary classifier
    primary_embeddings_attr_name = None
    if hasattr(primary_embedding_classifier, 'embeddings') and isinstance(primary_embedding_classifier.embeddings, torch.nn.Parameter):
        optimized_embeddings_data = primary_embedding_classifier.embeddings.data
        primary_embeddings_attr_name = 'embeddings'
    elif hasattr(primary_embedding_classifier, 'prototypes') and isinstance(primary_embedding_classifier.prototypes, torch.nn.Parameter):
        optimized_embeddings_data = primary_embedding_classifier.prototypes.data
    
    optimized_logit_scale_data = None
    if 'HierarchicalContrastiveLoss' in loss_type_from_config and \
       hasattr(primary_embedding_classifier, 'logit_scale') and \
       isinstance(primary_embedding_classifier.logit_scale, torch.nn.Parameter):
        optimized_logit_scale_data = primary_embedding_classifier.logit_scale.data

    for target_clf in all_embedding_classifiers_for_copy:
        if optimized_embeddings_data is not None:
            # Try to copy to the same attribute name as the primary, then try the other
            copied_to_attr = False
            if primary_embeddings_attr_name and hasattr(target_clf, primary_embeddings_attr_name) and \
               isinstance(getattr(target_clf, primary_embeddings_attr_name), torch.nn.Parameter):
                getattr(target_clf, primary_embeddings_attr_name).data.copy_(optimized_embeddings_data)
                copied_to_attr = True
            elif hasattr(target_clf, 'embeddings') and isinstance(target_clf.embeddings, torch.nn.Parameter):
                target_clf.embeddings.data.copy_(optimized_embeddings_data)
                copied_to_attr = True
            elif hasattr(target_clf, 'prototypes') and isinstance(target_clf.prototypes, torch.nn.Parameter):
                target_clf.prototypes.data.copy_(optimized_embeddings_data)
                copied_to_attr = True
            
            if not copied_to_attr:
                print(f"Warning: Could not copy embeddings to {type(target_clf).__name__} as it lacks .embeddings or .prototypes Parameter.")

        if optimized_logit_scale_data is not None:
            if hasattr(target_clf, 'logit_scale') and isinstance(target_clf.logit_scale, torch.nn.Parameter):
                target_clf.logit_scale.data.copy_(optimized_logit_scale_data)
            # else:
            #     print(f"Warning: Could not copy logit_scale to {type(target_clf).__name__} as it lacks .logit_scale Parameter or it was not optimized.")
    # --- End Copying ---
    
    checkpoint = {
        'meta': {
            'config_path': model_config_path,
            'pretrain_epochs': epochs,
            'config_text': cfg.text # Save the config text
        },
        'state_dict': model.state_dict() 
    }
    torch.save(checkpoint, output_checkpoint_file)
    print(f"Info: Pre-trained prototype checkpoint saved to {output_checkpoint_file}")
    return output_checkpoint_file

