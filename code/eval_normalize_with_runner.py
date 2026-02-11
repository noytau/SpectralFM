"""
Evaluate validation loss using EvaluationRunner._eval_validation_loss
Compares normalize=True vs normalize=False on sanity dataset.

Checkpoint: det_mask_seed42_100samples (trained with normalize=TRUE)
"""

import os
import sys
import torch

# Setup paths
sys.path.insert(0, '/mnt5/noy/SpectralFM/fairseq')
sys.path.insert(0, '/mnt5/noy/SpectralFM/code')
os.environ['USER_DIR'] = '/mnt5/noy/SpectralFM/fairseq/examples/data2vec'

from omegaconf import open_dict
from model_loader import load_fairseq_checkpoint
from evaluation_runner import EvaluationRunner


def run_evaluation():
    checkpoint_path = '/mnt5/noy/SpectralFM/checkpoints/runai/2026-01-27_20-17-21_det_mask_v6/checkpoint_best.pt'
    sanity_data_path = '/mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_100'
    
    print("=" * 70)
    print("VALIDATION LOSS COMPARISON using EvaluationRunner")
    print("=" * 70)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Dataset: {sanity_data_path} (sanity - 99 samples)")
    print("\n⚠️  TRAINING was done with normalize=TRUE (mask_seed=42)")
    print("=" * 70)
    
    # Create output directory for this evaluation
    output_dir = '/mnt5/noy/SpectralFM/code/eval_results/normalize_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for normalize_setting in [True, False]:
        print(f"\n{'='*60}")
        print(f"Evaluating with normalize={normalize_setting}")
        print(f"{'='*60}")
        
        # Load model fresh for each setting
        model, model_cfg, checkpoint_info = load_fairseq_checkpoint(checkpoint_path)
        cfg = checkpoint_info['cfg']
        
        # Modify normalize setting
        with open_dict(cfg):
            cfg.task.normalize = normalize_setting
            cfg.task.data = sanity_data_path
            cfg.common.fp16 = False  # Use fp32 for stable evaluation
        
        print(f"[+] task.normalize = {cfg.task.normalize}")
        
        # Create EvaluationRunner
        runner = EvaluationRunner(
            output_dir=output_dir,
            data_dir=sanity_data_path
        )
        
        # Run validation loss evaluation
        # Use split="sanity" to evaluate on training data (since valid.tsv may be small)
        metrics = runner._eval_validation_loss(
            model=model,
            cfg=cfg,
            split="valid",  # Use valid split
            eval_data_dir=sanity_data_path,
            debug=False
        )
        
        results[f'normalize={normalize_setting}'] = metrics
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTraining Config:     normalize = TRUE (with mask_seed=42)")
    print(f"Evaluation Dataset:  sanity (single_channel_100)")
    print()
    
    loss_true = results.get('normalize=True', {}).get('eval_loss', float('nan'))
    loss_false = results.get('normalize=False', {}).get('eval_loss', float('nan'))
    
    print(f"┌─────────────────────┬──────────────────┬────────────┐")
    print(f"│ Evaluation Setting  │ Validation Loss  │ Match?     │")
    print(f"├─────────────────────┼──────────────────┼────────────┤")
    print(f"│ normalize=True      │ {loss_true:16.6f} │ ✓ MATCHED  │")
    print(f"│ normalize=False     │ {loss_false:16.6f} │ ✗ MISMATCH │")
    print(f"└─────────────────────┴──────────────────┴────────────┘")
    print()
    
    if loss_true != float('nan') and loss_false != float('nan'):
        diff = loss_false - loss_true
        pct_diff = (diff / loss_true) * 100 if loss_true > 0 else 0
        print(f"Difference: {diff:+.6f} ({pct_diff:+.1f}%)")
        print()
        
        if loss_true < loss_false:
            print("✓ As expected: Loss is LOWER when evaluation matches training (normalize=True)")
        else:
            print("⚠️  Unexpected: Loss is higher when evaluation matches training")
    
    print("=" * 70)
    
    # Print full metrics
    print("\nFull Metrics:")
    for setting, metrics in results.items():
        print(f"\n{setting}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
    
    return results


if __name__ == '__main__':
    run_evaluation()
