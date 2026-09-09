import argparse
import torch
from mmengine.config import Config
from mmdet.registry import MODELS
from mmengine.registry import OPTIM_WRAPPERS, OPTIMIZERS
from mmdet.utils import register_all_modules
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
register_all_modules()

from hod.training.prototypes import perform_prototype_pretraining

def main():
    parser = argparse.ArgumentParser(
        description="Pre-train prototype embeddings based on class hierarchy, using a model config."
    )
    parser.add_argument('config', type=str, help="Path to the model config file (e.g., configs/dino/your_model_config.py).")
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help="Path to save the pre-trained model checkpoint (file or directory). "
             "If a directory, 'pretrained_prototypes.pth' is used. "
             "Defaults to 'work_dirs/{config_name}_prototype_pretrain/pretrained_prototypes.pth'."
    )
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use for training (e.g., 'cuda', 'cpu').")

    args = parser.parse_args()

    # Determine output file path
    if args.output_path is None:
        config_name = Path(args.config).stem
        output_dir = Path('work_dirs') / f"{config_name}_prototype_pretrain"
        output_file = output_dir / 'pretrained_prototypes.pth'
    else:
        output_file = Path(args.output_path)
        if output_file.is_dir():
            output_file = output_file / 'pretrained_prototypes.pth'

    try:
        saved_checkpoint_path = perform_prototype_pretraining(
            model_config_path=args.config,
            output_checkpoint_file=output_file,
            epochs=args.epochs,
            device=args.device
        )
        print(f"Standalone pre-training complete. Checkpoint at: {saved_checkpoint_path}")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error during prototype pre-training: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
