import argparse
import torch
# import torch.optim as optim # No longer directly used, will use MMEngine's builder
from mmengine.config import Config
# from mmengine.fileio import load # Removed as it was unused
from mmdet.registry import MODELS # For building models and components
from mmengine.registry import OPTIM_WRAPPERS, OPTIMIZERS # <-- Added OPTIMIZERS
from mmdet.utils import register_all_modules
register_all_modules()


# If your custom modules are not automatically registered, ensure they are imported.
# e.g., from hod.models.layers import EmbeddingClassifier
# from hod.models.losses import HierarchicalContrastiveLoss, EntailmentConeLoss
# These imports might be necessary if they are not picked up by MMDET's registry scanning.
# For now, assuming they are registered via __init__.py files in their respective modules.

# import math # Removed as it was unused
import sys
from pathlib import Path

# Add project root to sys.path to allow importing hod modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

def main():
    parser = argparse.ArgumentParser(
        description="Pre-train prototype embeddings based on class hierarchy, using a model config."
    )
    parser.add_argument('config', type=str, help="Path to the model config file (e.g., configs/dino/your_model_config.py).") # Made positional
    parser.add_argument(
        '--output_path', 
        type=str, 
        default=None, # Made optional, default is None
        help="Path to save the pre-trained model checkpoint (file or directory). "
             "If a directory is given, 'pretrained_prototypes.pth' will be used as filename. "
             "If not provided, defaults to 'work_dirs/{config_name}_prototype_pretrain/pretrained_prototypes.pth'."
    )
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    # parser.add_argument('--lr', type=float, default=0.01, help="Learning rate.") # Removed, will use config
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use for training (e.g., 'cuda', 'cpu').")

    args = parser.parse_args()

    # Determine output file path
    if args.output_path is None:
        # Create default path in work_dirs
        config_name = Path(args.config).stem
        output_dir = Path('work_dirs') / f"{config_name}_prototype_pretrain"
        output_file_path = output_dir / 'pretrained_prototypes.pth'
    else:
        output_file_path = Path(args.output_path)
        if output_file_path.is_dir(): # If user provided a directory
            output_file_path = output_file_path / 'pretrained_prototypes.pth'
    
    # Ensure parent directory exists
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Load configuration
    try:
        cfg = Config.fromfile(args.config)
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        sys.exit(1)

    # Set device in config or ensure model is moved to device
    # Note: cfg.device is not a standard MMEngine/MMDetection practice for model/data device placement.
    # Models are typically moved to device after creation, and data loaders handle device for data.
    # This line is kept for now as it was in the script, but consider aligning with MMEngine's device handling.
    # For this script, explicit .to(args.device) is used for model and tensors.
    # cfg.device = args.device 

    # Build the full model from config
    try:
        model = MODELS.build(cfg.model).to(args.device)
        model.train() # Set model to training mode
    except Exception as e:
        print(f"Error building model from config: {e}")
        sys.exit(1)

    # Access the EmbeddingClassifier
    embedding_classifier = None
    try:
        # Path based on user feedback and common DINO structure: model.bbox_head.cls_branches[0]
        if hasattr(model, 'bbox_head') and \
           hasattr(model.bbox_head, 'cls_branches') and \
           isinstance(model.bbox_head.cls_branches, torch.nn.ModuleList) and \
           len(model.bbox_head.cls_branches) > 0:
            
            branch_module = model.bbox_head.cls_branches[0] # Assuming we target the first branch
            
            # Check if the branch itself is an EmbeddingClassifier
            if type(branch_module).__name__ == 'EmbeddingClassifier':
                embedding_classifier = branch_module
                print("Info: Found EmbeddingClassifier at model.bbox_head.cls_branches[0].")
            # Check if the branch contains an fc_cls that is an EmbeddingClassifier
            elif hasattr(branch_module, 'fc_cls') and type(branch_module.fc_cls).__name__ == 'EmbeddingClassifier':
                embedding_classifier = branch_module.fc_cls
                print("Info: Found EmbeddingClassifier at model.bbox_head.cls_branches[0].fc_cls.")
            else:
                print(f"Warning: Module at model.bbox_head.cls_branches[0] (or its fc_cls) is not an EmbeddingClassifier. Found type: {type(branch_module).__name__}.")

        # Fallback for DINO-style heads where fc_cls might be directly on bbox_head (if no cls_branches)
        if embedding_classifier is None and hasattr(model, 'bbox_head') and \
           hasattr(model.bbox_head, 'fc_cls') and \
           type(model.bbox_head.fc_cls).__name__ == 'EmbeddingClassifier':
            embedding_classifier = model.bbox_head.fc_cls
            print("Info: Found EmbeddingClassifier at model.bbox_head.fc_cls.")
        
        # Fallback if bbox_head itself is the classifier (less common for DINO but possible for simpler custom heads)
        if embedding_classifier is None and hasattr(model, 'bbox_head') and \
            type(model.bbox_head).__name__ == 'EmbeddingClassifier':
            embedding_classifier = model.bbox_head
            print("Info: Found EmbeddingClassifier at model.bbox_head.")

        if embedding_classifier is None:
            print("Error: Could not automatically find EmbeddingClassifier. Please check model structure and script.")
            print("Attempted paths like model.bbox_head.cls_branches[0], model.bbox_head.cls_branches[0].fc_cls, and model.bbox_head.fc_cls.")
            sys.exit(1)
        
        print(f"Using EmbeddingClassifier: {type(embedding_classifier).__name__}")

    except AttributeError as e:
        print(f"Error accessing EmbeddingClassifier in the model structure: {e}. Check config and model definition.")
        sys.exit(1)

    # Build the loss function from config
    loss_embed_cfg = None # Renamed from loss_cls_cfg
    try:
        # Try to get loss from the branch config first (e.g., model.bbox_head.cls_branches[0].loss_embed)
        if hasattr(cfg.model, 'bbox_head') and \
           hasattr(cfg.model.bbox_head, 'cls_branches') and \
           isinstance(cfg.model.bbox_head.cls_branches, list) and \
           len(cfg.model.bbox_head.cls_branches) > 0 and \
           'loss_embed' in cfg.model.bbox_head.cls_branches[0]: # Changed from loss_cls
            loss_embed_cfg = cfg.model.bbox_head.cls_branches[0].loss_embed # Changed from loss_cls
            print("Info: Using loss_embed from config at model.bbox_head.cls_branches[0].loss_embed.")
        
        # Fallback to bbox_head level loss_embed (common if shared or no branches)
        elif hasattr(cfg.model, 'bbox_head') and \
             'loss_embed' in cfg.model.bbox_head: # Changed from loss_cls
            loss_embed_cfg = cfg.model.bbox_head.loss_embed # Changed from loss_cls
            print("Info: Using loss_embed from config at model.bbox_head.loss_embed.")
        
        if loss_embed_cfg is None:
            print("Error: Could not find loss_embed configuration in expected locations (e.g., model.bbox_head.cls_branches[0].loss_embed or model.bbox_head.loss_embed).") # Updated message
            sys.exit(1)
            
        loss_fn = MODELS.build(loss_embed_cfg).to(args.device) # Using loss_embed_cfg
        print(f"Successfully built loss function: {loss_embed_cfg.type}")

    except Exception as e: # Catching general exception for robust error reporting during setup
        print(f"Error building loss function from config: {e}")
        sys.exit(1)

    # Determine parameters to optimize
    params_to_optimize = []
    if hasattr(embedding_classifier, 'embeddings') and isinstance(embedding_classifier.embeddings, torch.nn.Parameter):
        params_to_optimize.append(embedding_classifier.embeddings)
    else:
        print("Error: embedding_classifier.embeddings not found or not a Parameter. Cannot optimize.")
        sys.exit(1)

    loss_type_from_config = loss_embed_cfg.type # Using loss_embed_cfg
    if 'HierarchicalContrastiveLoss' in loss_type_from_config:
        if hasattr(embedding_classifier, 'logit_scale') and isinstance(embedding_classifier.logit_scale, torch.nn.Parameter):
            params_to_optimize.append(embedding_classifier.logit_scale)
        else:
            print("Warning: HierarchicalContrastiveLoss is used, but logit_scale not found or not a Parameter in EmbeddingClassifier. It will not be optimized.")
    
    if not params_to_optimize:
        print("Error: No parameters to optimize. Exiting.")
        sys.exit(1)

    # Build OptimWrapper from config
    optim_wrapper_instance = None
    try:
        if not hasattr(cfg, 'optim_wrapper'):
            print("Error: Optimizer wrapper configuration (cfg.optim_wrapper) not found in config.")
            sys.exit(1)
        
        # 1. Prepare the optimizer_config
        import copy # Ensure copy is imported
        optimizer_config_source = cfg.optim_wrapper.get('optimizer')
        if optimizer_config_source is None:
            print("Error: cfg.optim_wrapper.optimizer key is missing.")
            sys.exit(1)
        if not isinstance(optimizer_config_source, (dict, Config)): # ConfigDict is a subclass of Config, Config is dict-like
            print(f"Error: cfg.optim_wrapper.optimizer is not a dictionary-like object, got {type(optimizer_config_source)}.")
            sys.exit(1)
            
        optimizer_config = copy.deepcopy(dict(optimizer_config_source)) # Convert to plain dict and deepcopy

        # Inject the specific parameters to optimize
        optimizer_config['params'] = params_to_optimize

        # 2. Build the actual optimizer instance
        actual_optimizer = OPTIMIZERS.build(optimizer_config)
        
        # 3. Prepare the OptimWrapper config, using the instantiated optimizer
        # Start with a copy of the original optim_wrapper config
        optim_wrapper_base_cfg = cfg.optim_wrapper.copy() 
        
        # Remove paramwise_cfg if it exists, as it's handled by DefaultOptimWrapperConstructor usually
        # and not directly by OptimWrapper.__init__ if we provide an instantiated optimizer.
        if 'paramwise_cfg' in optim_wrapper_base_cfg:
            del optim_wrapper_base_cfg['paramwise_cfg']
            print("Info: Removed 'paramwise_cfg' from optim_wrapper_base_cfg for this script.")
            
        # Replace the 'optimizer' (which was a config dict) with the actual instance
        optim_wrapper_base_cfg['optimizer'] = actual_optimizer 
        
        # 4. Build the OptimWrapper
        optim_wrapper_instance = OPTIM_WRAPPERS.build(optim_wrapper_base_cfg)
        
        effective_lr = optimizer_config.get('lr', 'N/A')
        # Ensure optimizer_config['type'] exists before accessing
        optimizer_type_str = optimizer_config.get('type', 'N/A')
        print(f"Successfully built OptimWrapper with optimizer: {optimizer_type_str} with LR: {effective_lr}")

    except Exception as e:
        print(f"Error building OptimWrapper from config: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        sys.exit(1)

    print(f"Starting pre-training of prototypes for {args.epochs} epochs on device '{args.device}'.")
    print(f"Using loss: {loss_type_from_config}")
    print(f"Optimizing {len(params_to_optimize)} parameter group(s).")


    for epoch in range(args.epochs):
        # Use OptimWrapper methods
        optim_wrapper_instance.optimizer.zero_grad() # Or just optim_wrapper_instance.zero_grad()

        current_prototypes = embedding_classifier.prototypes # Shape: (num_classes, embed_dims)
        loss = torch.tensor(0.0, device=args.device)

        if 'HierarchicalContrastiveLoss' in loss_type_from_config:
            if not hasattr(embedding_classifier, 'get_distance_logits'):
                print("Error: EmbeddingClassifier is missing the 'get_distance_logits' method, which is required for HCL pre-training in this script.")
                sys.exit(1)
            
            # current_prototypes are embedding_classifier.prototypes.
            # Calling get_distance_logits(current_prototypes) will compute scaled negative distances
            # between current_prototypes (as features) and the classifier's own prototypes (which are current_prototypes).
            # This effectively gives the pairwise prototype interaction matrix needed for HCL.
            # The geometric_bias potentially added by get_distance_logits is fine, as HCL is invariant to it.
            logits_matrix = embedding_classifier.get_distance_logits(current_prototypes.unsqueeze(0), current_prototypes)
            loss = loss_fn(logits_matrix.squeeze(0)) # HCL expects (logits_matrix)

        elif 'EntailmentConeLoss' in loss_type_from_config:
            # ECL operates directly on the prototypes
            loss = loss_fn(current_prototypes) 
        
        else:
            print(f"Error: Loss type {loss_type_from_config} from config is not supported by this script's training loop.")
            sys.exit(1)

        loss.backward()
        optim_wrapper_instance.step() # This will call optimizer.step() and handle grad clipping if configured

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            log_msg = f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}"
            if 'HierarchicalContrastiveLoss' in loss_type_from_config and hasattr(embedding_classifier, 'logit_scale'):
                log_msg += f", LogitScale: {embedding_classifier.logit_scale.item():.4f}"
            print(log_msg)

    # Save the pre-trained model checkpoint (full model state_dict)
    # output_file_path is now defined earlier and parent directory is already created
    
    checkpoint = {
        'meta': {
            'config': cfg.text, # Save the config text for reference
            'epoch': args.epochs,
            'script_args': vars(args)
        },
        'state_dict': model.state_dict() # Save the entire model's state dictionary
    }
    torch.save(checkpoint, output_file_path) # Use the determined output_file_path
    print(f"Pre-trained model checkpoint saved to {output_file_path}") # Use the determined output_file_path
    print("Note: This checkpoint contains the full model state_dict, but only prototype-related parameters were trained.")

if __name__ == '__main__':
    main()
