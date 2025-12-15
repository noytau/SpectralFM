import sys
import os
import argparse
import torch
from fairseq import utils

# 1. Import the loader function
from load_fairseq_model import load_data2vec_model

def get_model_device(model):
    """Safely get the device of a Fairseq model."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        # Fallback if model has no parameters (unlikely)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_evaluation(args):
    """
    Combines Fairseq Loading with Custom EvalExperiment.
    """
    
    # ------------------------------------------------------------------
    # A. CONFIGURATION
    # ------------------------------------------------------------------
    TRAINING_DATA_PATH = "/mnt5/noy/fairseq/data/single_channel_1m/"
    EVAL_DATA_PATH = "/mnt5/noy/nova_samples/one_chnl/" 
    CUSTOM_CODE_PATH = "/mnt5/noy/code"

    print(f"Model Init Context: {TRAINING_DATA_PATH}")
    print(f"Evaluation Data:    {EVAL_DATA_PATH}")

    # --- Verify Code Path ---
    if os.path.exists(CUSTOM_CODE_PATH):
        if CUSTOM_CODE_PATH not in sys.path:
            sys.path.insert(0, CUSTOM_CODE_PATH)
    else:
        print(f"❌ Custom code path NOT found: {CUSTOM_CODE_PATH}")
        return

    # ------------------------------------------------------------------
    # B. Load Model (Using TRAINING Context)
    # ------------------------------------------------------------------
    model, fairseq_cfg, task = load_data2vec_model(
        checkpoint_path=args.model_path,
        data_path_override=TRAINING_DATA_PATH, 
        user_dir="examples/data2vec",     
        extra_code_paths=[CUSTOM_CODE_PATH],
        cpu=False
    )
    
    # ------------------------------------------------------------------
    # C. ATTACH TO ARGS (The requested change)
    # ------------------------------------------------------------------
    print("\n--- Attaching Model and Task to args ---")
    
    # Attach model and task
    args.model = model
    args.task = task
    
    # PATCH: Add a .device property to the model if it's missing
    # Your evaluate.py code expects model.device to exist.
    if not hasattr(model, 'device'):
        device = get_model_device(model)
        print(f"Patching model.device to: {device}")
        model.device = device
    
    # Ensure model is in eval mode
    args.model.eval()

    # ------------------------------------------------------------------
    # D. INITIALIZE EVALUATOR
    # ------------------------------------------------------------------
    print("--- Initializing Custom EvalExperiment ---")
    
    try:
        from evaluate import EvalExperiment 
        
        # Now EvalExperiment(args) can access args.model inside its __init__
        evaluator = EvalExperiment(args)
        
        print("Running evaluator.run_evaluation()...")
        evaluator.run_evaluation()
        
    except ImportError as e:
        print("\n❌ IMPORT ERROR:")
        print(f"   {e}")
        print("   Check if 'evaluate.py' exists in /mnt5/noy/code/")
    except Exception as e:
        print(f"❌ Error during custom evaluation: {e}")
        import traceback
        traceback.print_exc()

# ------------------------------------------------------------------
# Simulation Block
# ------------------------------------------------------------------
if __name__ == "__main__":
    
    class MockArgs:
        def __init__(self):
            self.run_id = 1
            self.test_dir = '/mnt5/noy/nova_samples/one_chnl/'
            self.arch = 'conv1d'
            self.masking_type = 'span'
            self.mask_ratio = 0.15
            self.loss_function = 'mse'
            self.learning_rate = 0.0001
            self.batch_size = 16
            self.epoch = 1
            self.model_path = '/mnt5/noy/fairseq/outputs/2025-11-30/18-12-42/checkpoints/checkpoint_best.pt'
            self.model_name = 'checkpoint_las'
            self.output_path = '/mnt5/noy/code/outputs'
            self.mode = 'eval'
            self.eval_method = 'signal_completion'
            self.test_method = 'index_in_distribution_stack_holdout'
            self.stack_method = 'index'
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = MockArgs()
    run_evaluation(args)
