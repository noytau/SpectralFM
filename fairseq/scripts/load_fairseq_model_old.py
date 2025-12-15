import torch
import sys
import os
import argparse
from fairseq import checkpoint_utils, tasks, utils

# --- FIX: Add project root to sys.path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------------

def load_data2vec_model(checkpoint_path, data_path_override=None, user_dir=None, extra_code_paths=None, cpu=False):
    """
    Loads a fairseq Data2Vec model, task, and configuration from a checkpoint.
    """
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # 0. Register Custom Code Paths (CRITICAL for user-defined code)
    if extra_code_paths:
        for path in extra_code_paths:
            if os.path.exists(path):
                print(f"Adding custom code path: {path}")
                if path not in sys.path:
                    sys.path.insert(0, path)
            else:
                print(f"Warning: Custom code path does not exist: {path}")

    # 1. Import User Directory (CRITICAL for Data2Vec & Plugins)
    if user_dir:
        print(f"Importing user module from: {user_dir}")
        utils.import_user_module(argparse.Namespace(user_dir=user_dir))

    # 2. Prepare Argument Overrides
    arg_overrides = {}
    if data_path_override:
        arg_overrides['data'] = data_path_override
        arg_overrides['task'] = {'data': data_path_override}

    # 3. Load the Ensemble
    print(f"Loading model from {checkpoint_path}...")
    try:
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path],
            arg_overrides=arg_overrides,
            suffix='' 
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nPossible Fixes:")
        print("1. Use '--user-dir examples/data2vec' for data2vec models.")
        print("2. Use '--extra-code /path/to/code' if your model uses custom code outside fairseq.")
        raise e

    model = models[0]

    # 4. Move to Device
    if cpu:
        model.cpu()
    else:
        if torch.cuda.is_available():
            model.cuda()
    
    # 5. Set to Evaluation Mode
    model.eval()

    return model, cfg, task

# --- Example Usage Block ---
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', help='Path to checkpoint file')
    parser.add_argument('--data', help='Override data path', default=None)
    parser.add_argument('--user-dir', help='Path to user directory (e.g. examples/data2vec)', default=None)
    parser.add_argument('--extra-code', help='Path to custom code folder (e.g. /mnt5/noy/code)', default=None)
    args = parser.parse_args()

    # Smart Defaults
    if not args.user_dir and os.path.exists('examples/data2vec'):
        print("Tip: Auto-detected 'examples/data2vec'. You might want to add '--user-dir examples/data2vec'")

    # Convert single path string to list
    extra_paths = [args.extra_code] if args.extra_code else []

    # 1. Load the Model
    model, cfg, task = load_data2vec_model(
        args.ckpt, 
        data_path_override=args.data,
        user_dir=args.user_dir,
        extra_code_paths=extra_paths
    )
    
    print("\n✅ Model Loaded Successfully")
    print(f"Model Class: {model.__class__.__name__}")

    # ==========================================
    # 2. RUN YOUR CUSTOM EVALUATION HERE
    # ==========================================
    print("\n--- Starting Custom Evaluation ---")
    
    # Example:
    # Assuming your code is in /mnt5/noy/code/my_eval_script.py
    # and has a function called 'evaluate_model(model, task)'
    
    try:
        # Import your module (Python can find it because we added --extra-code path above)
        from evaluate import EvalExperiment  
        
        # Run your function
        print("Calling evaluator")
        evaluator = EvalExperiment() # fixme pass args 
        evaluator.run_evaluation()

    except ImportError:
        print("⚠️  Could not import 'my_eval_script'.")
        print(f"   Make sure your python file is inside: {args.extra_code}")
        print("   And that you are importing the correct filename.")
    except AttributeError:
        print("⚠️  Imported module, but could not find function 'evaluate_model'.")
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
