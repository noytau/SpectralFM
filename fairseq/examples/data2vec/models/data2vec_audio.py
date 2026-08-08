# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

from omegaconf import II

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq.modules import EMAModule, EMAModuleConfig
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec import (
    ConvFeatureExtractionModel,
    Wav2Vec2Config,
    TransformerEncoder,
)
from fairseq.modules import (
    GradMultiply,
    LayerNorm,
)
from fairseq.utils import index_put
from fairseq import checkpoint_utils


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-component init / freeze / param-group tagging  (this PR)
#
#  Defers to ``code/recon_components.py`` for the actual shape-safe loaders
#  (auto-detects fairseq_audio / data2vec_multi / apr28_fe_recon layouts and
#  remaps ViT ``blocks.X.*`` → fairseq ``layers.X.*`` with QKV split).
#
#  The module is imported lazily so this file stays importable even when run
#  outside the SpectralFM workspace (e.g. clean fairseq install). Add the repo's
#  ``code/`` directory to ``PYTHONPATH`` (or pass it via runai submit env) to
#  enable the per-component init/freeze/tagging fields.
# ─────────────────────────────────────────────────────────────────────────────

# Top-level dotted-name prefix → composite-optimizer group name. Encoder and
# decoder of the same modality share a group per user spec; LN / proj are their
# own groups so they can run at a higher LR than the pretrained backbone.
_PARAM_GROUP_MAP: Dict[str, str] = {
    "feature_extractor":    "fe",
    "fe_recon_decoder":     "fe",
    "layer_norm":           "ln",
    "post_extract_proj":    "proj",
    "encoder":              "transformer",
    "trans_recon_decoder":  "transformer",
}


def _try_import_recon_components():
    try:
        import recon_components as rc  # type: ignore
        return rc
    except Exception as e:
        logger.warning(
            "recon_components could not be imported (%s); "
            "init_*_ckpt / freeze_*_v2 / tag_param_groups will be no-ops.", e,
        )
        return None


def _maybe_apply_recon_components(model, cfg) -> None:
    """Apply the per-component init / freeze / param-group tagging requested in cfg.

    No-op when no field is set, so this PR is fully back-compat with existing
    YAML configs (every new field defaults to None/False).
    """
    any_init = any(
        bool(getattr(cfg, k, None))
        for k in ("init_fe_ckpt", "init_ln_ckpt", "init_proj_ckpt",
                  "init_transformer_ckpt", "init_fe_recon_decoder_ckpt",
                  "init_trans_recon_decoder_ckpt")
    )
    _FREEZE_KEYS = ("freeze_fe_v2", "freeze_ln", "freeze_proj",
                    "freeze_transformer_v2", "freeze_fe_recon_decoder",
                    "freeze_trans_recon_decoder", "freeze_mask_emb",
                    "freeze_final_proj")
    any_freeze = any(getattr(cfg, k, None) is not None for k in _FREEZE_KEYS)
    tag = bool(getattr(cfg, "tag_param_groups", False))
    if not (any_init or any_freeze or tag):
        return

    rc = _try_import_recon_components() if any_init else None

    if any_init and rc is not None:
        logger.info("[recon_components] per-component init from cfg.init_*_ckpt:")
        for name, attr, loader_name, target in (
            ("fe",                  "init_fe_ckpt",                  "load_fe_from_ckpt",                 model),
            ("ln",                  "init_ln_ckpt",                  "load_layer_norm_from_ckpt",         model),
            ("proj",                "init_proj_ckpt",                "load_post_extract_proj_from_ckpt",  model),
            ("transformer",         "init_transformer_ckpt",         "load_transformer_from_ckpt",        model),
            ("fe_recon_decoder",    "init_fe_recon_decoder_ckpt",    "load_head_from_ckpt",               model.fe_recon_decoder),
            ("trans_recon_decoder", "init_trans_recon_decoder_ckpt", "load_head_from_ckpt",               model.trans_recon_decoder),
        ):
            p = (getattr(cfg, attr, None) or "").strip()
            if not p or p.lower() == "none":
                continue
            getattr(rc, loader_name)(target, p)

    if any_freeze:
        # Semantics (since the "explicit-unfreeze" patch):
        #   True  → freeze the component (requires_grad=False; FE → feature_grad_mult=0)
        #   False → UNFREEZE the component, overriding any prior freezing
        #           (e.g. cfg.train_only_fe=True → freeze_all_except_feature_extractor()).
        #   None  → leave whatever requires_grad state the module already has.
        # This lets a recon_loss YAML say `freeze_transformer_v2: false` to force the
        # transformer to be trainable even if some earlier code path froze it.
        logger.info("[recon_components] per-component freeze (True=freeze, False=unfreeze, None=no-op):")
        freezes = {
            "feature_extractor":    getattr(cfg, "freeze_fe_v2", None),
            "layer_norm":           getattr(cfg, "freeze_ln", None),
            "post_extract_proj":    getattr(cfg, "freeze_proj", None),
            "encoder":              getattr(cfg, "freeze_transformer_v2", None),
            "fe_recon_decoder":     getattr(cfg, "freeze_fe_recon_decoder", None),
            "trans_recon_decoder":  getattr(cfg, "freeze_trans_recon_decoder", None),
        }
        # FE needs special handling because freezing only the params isn't enough;
        # SpectralFM also scales FE grads via `feature_grad_mult` upstream of the FE.
        # Remember the user-configured value so we can restore it on explicit-unfreeze.
        cfg_feature_grad_mult = float(getattr(cfg, "feature_grad_mult", 1.0))

        for attr, do_freeze in freezes.items():
            if do_freeze is None:
                continue
            mod = getattr(model, attr, None)
            if mod is None:
                continue
            want_grad = not bool(do_freeze)
            n_changed = 0
            for p in mod.parameters():
                if p.requires_grad != want_grad:
                    p.requires_grad = want_grad
                    n_changed += p.numel()
            verb = "froze" if do_freeze else "unfroze"
            logger.info(f"  {attr}: {verb} {n_changed:,} params (now requires_grad={want_grad})")
            if attr == "feature_extractor":
                if do_freeze:
                    model.feature_grad_mult = 0.0
                    logger.info("  feature_extractor: also set feature_grad_mult=0")
                else:
                    model.feature_grad_mult = cfg_feature_grad_mult
                    logger.info(
                        f"  feature_extractor: restored feature_grad_mult={cfg_feature_grad_mult}"
                    )

        mask_emb_freeze = getattr(cfg, "freeze_mask_emb", None)
        if mask_emb_freeze is not None and hasattr(model, "mask_emb"):
            want_grad = not bool(mask_emb_freeze)
            if model.mask_emb.requires_grad != want_grad:
                model.mask_emb.requires_grad = want_grad
                verb = "froze" if mask_emb_freeze else "unfroze"
                logger.info(f"  mask_emb: {verb} 1 param (now requires_grad={want_grad})")

        final_proj_freeze = getattr(cfg, "freeze_final_proj", None)
        if final_proj_freeze is not None and getattr(model, "final_proj", None) is not None:
            want_grad = not bool(final_proj_freeze)
            n_changed = 0
            for p in model.final_proj.parameters():
                if p.requires_grad != want_grad:
                    p.requires_grad = want_grad
                    n_changed += p.numel()
            if n_changed:
                verb = "froze" if final_proj_freeze else "unfroze"
                logger.info(
                    f"  final_proj: {verb} {n_changed:,} params (now requires_grad={want_grad})"
                )

    if tag:
        n_tagged: Dict[str, int] = {}
        for name, p in model.named_parameters():
            top = name.split(".", 1)[0]
            group = _PARAM_GROUP_MAP.get(top, "other")
            p.param_group = group  # consumed by fairseq.optim.composite
            n_tagged[group] = n_tagged.get(group, 0) + 1
        logger.info(f"[recon_components] tagged param_group → tensor counts: {n_tagged}")


def build_post_extract_proj(
    in_dim: int, out_dim: int, proj_type: str, mlp_hidden: int
) -> nn.Module:
    """FE channel space → transformer embed dim (cf. wav2vec2 post_extract_proj).

    See fairseq/models/wav2vec/wav2vec2.py (conditional Linear when embed !=
    encoder_embed_dim) and Data2VecAudioModel below.
    """
    t = (proj_type or "linear").strip().lower()
    if t == "mlp_gelu":
        h = max(1, int(mlp_hidden))
        return nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Linear(h, out_dim),
        )
    if t != "linear":
        logger.warning(
            "Unknown post_extract_proj_type=%r; using linear", proj_type
        )
    return nn.Linear(in_dim, out_dim)


# Debug plotting configuration
_DEBUG_PLOT_DIR = None
_DEBUG_PLOT_MODE = None  # 'train' or 'eval'
_DEBUG_ENABLED = False  # Global flag to enable/disable debug plots
_DEBUG_INPUT_COUNTER = 0
_DEBUG_PRED_COUNTER = 0
_DEBUG_INPUT_MAX_SAMPLES = 20


def set_debug_plot_mode(mode: str, output_dir: str = None):
    """Set the debug plotting mode and optionally custom output directory.
    
    Args:
        mode: 'train' or 'eval' - this creates a subdirectory
        output_dir: Optional custom base directory. If None, uses default.
    """
    global _DEBUG_PLOT_DIR, _DEBUG_PLOT_MODE, _DEBUG_ENABLED, _DEBUG_INPUT_COUNTER, _DEBUG_PRED_COUNTER
    import os
    from datetime import datetime
    
    _DEBUG_ENABLED = True
    _DEBUG_PLOT_MODE = mode
    _DEBUG_INPUT_COUNTER = 0
    _DEBUG_PRED_COUNTER = 0
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir:
        _DEBUG_PLOT_DIR = os.path.join(output_dir, f'debug_plots_{mode}_{timestamp}')
    else:
        _DEBUG_PLOT_DIR = f'/mnt5/noy/SpectralFM/code/eval_results/debug_{mode}_{timestamp}'
    
    os.makedirs(_DEBUG_PLOT_DIR, exist_ok=True)
    print(f"Debug plots ({mode}) will be saved to: {_DEBUG_PLOT_DIR}")


def _get_debug_plot_dir():
    """Get or create the debug plot directory with datetime.
    Only creates directory and prints message if debug mode is enabled.
    """
    global _DEBUG_PLOT_DIR, _DEBUG_PLOT_MODE, _DEBUG_ENABLED
    if not _DEBUG_ENABLED:
        return None
    if _DEBUG_PLOT_DIR is None:
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode = _DEBUG_PLOT_MODE or 'unknown'
        _DEBUG_PLOT_DIR = f'/mnt5/noy/SpectralFM/code/eval_results/debug_{mode}_{timestamp}'
        os.makedirs(_DEBUG_PLOT_DIR, exist_ok=True)
        print(f"Debug plots will be saved to: {_DEBUG_PLOT_DIR}")
    return _DEBUG_PLOT_DIR


def debug_plot_input_masking(source, features_before_mask, mask_indices, x_after_mask, sample_indices=None):
    """
    Plot input space and masking visualization (4 subplots) for all samples in batch.
    
    Args:
        source: Raw input spectrogram [B, T_raw] or [B, C, T_raw]
        features_before_mask: Features before masking [B, T, C]
        mask_indices: Boolean mask [B, T]
        x_after_mask: Features after masking applied [B, T, C]
        sample_indices: Optional tensor of dataset sample indices [B]
    
    Runs on first 20 samples total (across all batches).
    """
    global _DEBUG_INPUT_COUNTER, _DEBUG_ENABLED
    
    if not _DEBUG_ENABLED:
        return
    
    if _DEBUG_INPUT_COUNTER >= _DEBUG_INPUT_MAX_SAMPLES:
        return
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    save_dir = _get_debug_plot_dir()
    if save_dir is None:
        return
    input_dir = os.path.join(save_dir, 'input_masking')
    os.makedirs(input_dir, exist_ok=True)
    
    batch_size = source.shape[0]
    
    # Loop over all samples in the batch
    for i in range(batch_size):
        if _DEBUG_INPUT_COUNTER >= _DEBUG_INPUT_MAX_SAMPLES:
            return
        
        _DEBUG_INPUT_COUNTER += 1
        # Use actual sample index from dataset if available, otherwise use counter
        if sample_indices is not None:
            dataset_idx = int(sample_indices[i].item()) if hasattr(sample_indices[i], 'item') else int(sample_indices[i])
            sample_id = f'idx{dataset_idx:04d}'
        else:
            sample_id = f'seq{_DEBUG_INPUT_COUNTER:03d}'
        
        # Extract sample i from batch
        src_np = source[i].detach().cpu().float().numpy()  # [T_raw] or [C, T_raw]
        feat_before_np = features_before_mask[i].detach().cpu().float().numpy()  # [T, C]
        feat_after_np = x_after_mask[i].detach().cpu().float().numpy()  # [T, C]
        m = mask_indices[i].cpu().numpy()  # [T]
        
        n_masked = int(m.sum())
        n_total = len(m)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Input spectrogram (1D waveform plot)
        ax1 = axes[0, 0]
        if src_np.ndim == 1:
            ax1.plot(src_np, linewidth=0.5)
            ax1.set_title(f'Input waveform - length: {len(src_np)}')
        else:
            # If 2D, plot as heatmap
            ax1.imshow(src_np, aspect='auto', cmap='viridis')
            ax1.set_title(f'Input spectrogram - shape: {src_np.shape}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude' if src_np.ndim == 1 else 'Channel')
        
        # 2. Features before masking (2D heatmap)
        ax2 = axes[0, 1]
        im2 = ax2.imshow(feat_before_np.T, aspect='auto', cmap='viridis')
        ax2.set_title(f'Features BEFORE mask - shape: {feat_before_np.shape}')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Feature dim')
        plt.colorbar(im2, ax=ax2)
        
        # 3. Mask indicator
        ax3 = axes[1, 0]
        ax3.imshow(m.reshape(1, -1), aspect='auto', cmap='Reds', interpolation='nearest')
        ax3.set_title(f'MASK: {n_masked}/{n_total} positions masked ({100*n_masked/n_total:.1f}%)')
        ax3.set_yticks([])
        ax3.set_xlabel('Time')
        
        # 4. Features after masking (2D heatmap)
        ax4 = axes[1, 1]
        im4 = ax4.imshow(feat_after_np.T, aspect='auto', cmap='viridis')
        ax4.set_title(f'Features AFTER mask - shape: {feat_after_np.shape}')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Feature dim')
        plt.colorbar(im4, ax=ax4)
        
        # Add title with sample index
        if sample_indices is not None:
            dataset_idx = int(sample_indices[i].item()) if hasattr(sample_indices[i], 'item') else int(sample_indices[i])
            fig.suptitle(f'Sample: Dataset Index {dataset_idx}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(input_dir, f'sample_{sample_id}.png')
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        
        idx_info = f" (dataset_idx={dataset_idx})" if sample_indices is not None else ""
        print(f"[Input {_DEBUG_INPUT_COUNTER}/{_DEBUG_INPUT_MAX_SAMPLES}] Masked: {n_masked}/{n_total}{idx_info} | saved to {save_path}")


def debug_plot_predictions(x_pred, y_target):
    """
    Plot prediction comparison (3 subplots: x_pred, y_target, diff).
    
    Args:
        x_pred: Prediction after final_proj [N_masked, C]
        y_target: EMA target [N_masked, C]
    
    Runs on ALL batches (no limit).
    """
    global _DEBUG_PRED_COUNTER, _DEBUG_ENABLED
    
    if not _DEBUG_ENABLED:
        return
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    save_dir = _get_debug_plot_dir()
    if save_dir is None:
        return
    pred_dir = os.path.join(save_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    
    _DEBUG_PRED_COUNTER += 1
    batch_id = f'{_DEBUG_PRED_COUNTER:03d}'
    
    # Convert to numpy
    x_np = x_pred.detach().cpu().float().numpy()  # [N_masked, C]
    y_np = y_target.detach().cpu().float().numpy()  # [N_masked, C]
    diff_np = x_np - y_np
    mse = (diff_np**2).mean()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. x_pred
    im1 = axes[0].imshow(x_np.T, aspect='auto', cmap='viridis')
    axes[0].set_title(f'x_pred - shape: {x_np.shape}')
    axes[0].set_xlabel('Masked position idx')
    axes[0].set_ylabel('Feature dim')
    plt.colorbar(im1, ax=axes[0])
    
    # 2. y_target
    im2 = axes[1].imshow(y_np.T, aspect='auto', cmap='viridis')
    axes[1].set_title(f'y_target (EMA) - shape: {y_np.shape}')
    axes[1].set_xlabel('Masked position idx')
    axes[1].set_ylabel('Feature dim')
    plt.colorbar(im2, ax=axes[1])
    
    # 3. diff
    im3 = axes[2].imshow(diff_np.T, aspect='auto', cmap='RdBu_r')
    axes[2].set_title(f'x_pred - y_target | MSE: {mse:.6f}')
    axes[2].set_xlabel('Masked position idx')
    axes[2].set_ylabel('Feature dim')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    save_path = os.path.join(pred_dir, f'batch_{batch_id}.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


@dataclass
class Data2VecAudioConfig(Wav2Vec2Config):

    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )
    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )

    layer_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False
    batch_norm_target_layer: bool = False
    group_norm_target_layer: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_transformer_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer"},
    )
    ema_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )
    train_only_fe: bool = field(
        default=True, metadata={"help": "Train only feature-extractor, freeze other parts"}
    )

    max_update: int = II("optimization.max_update")

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )
    
    # Mask saving for loss validation
    mask_save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save mask indices during training for loss validation"},
    )
    
    # Mask memory saving (for fixed masking experiments)
    save_mask_memory: bool = field(
        default=False,
        metadata={"help": "If True, enable mask memory saving during validation. Masks will be stored and saved for later reuse in evaluation."},
    )
    
    mask_memory_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save mask memory after validation. If save_mask_memory=True and this is None, will auto-generate path based on checkpoint.save_dir."},
    )
    
    # Deterministic masking seed
    mask_seed: Optional[int] = field(
        default=None,
        metadata={
            "help": "Base seed for deterministic masking. When set, masks are computed as "
                    "hash(seed, 0, sample_id), ensuring each sample always gets the same mask "
                    "across training and evaluation. If None, random masking is used."
        },
    )
    
    # Debug plotting
    debug_plot_masking: bool = field(
        default=False,
        metadata={"help": "If True, enable debug plotting of input masking visualizations. Plots are saved to eval_results/debug_*/input_masking/"},
    )

    model_path: Optional[str] = field(default=None)
    skip_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )

    # --- Tier 1 ablation fields ---
    fe_mode: str = field(
        default="train",
        metadata={"help": "Feature extractor mode: 'train' | 'frozen' | 'identity'"},
    )
    freeze_encoder: bool = field(
        default=False,
        metadata={"help": "Freeze all transformer encoder parameters"},
    )
    post_extract_proj_type: str = field(
        default="linear",
        metadata={
            "help": "FE→transformer map: 'linear' (single Linear) or "
            "'mlp_gelu' (Linear → GELU → Linear to encoder_embed_dim)"
        },
    )
    post_extract_proj_mlp_hidden: int = field(
        default=1536,
        metadata={"help": "Hidden size when post_extract_proj_type=mlp_gelu"},
    )
    lambda_var: float = field(
        default=0.0,
        metadata={"help": "Weight for variance regularization loss (VICReg-style)"},
    )
    lambda_cov: float = field(
        default=0.0,
        metadata={"help": "Weight for covariance regularization loss (VICReg-style)"},
    )
    lambda_uniform: float = field(
        default=0.0,
        metadata={"help": "Weight for uniformity loss (Wang & Isola 2020)"},
    )
    var_gamma: float = field(
        default=1.0,
        metadata={"help": "Target std for variance hinge loss"},
    )

    # --- Reconstruction loss (optional; EMA regression unchanged) ---
    recon_only: bool = field(
        default=False,
        metadata={"help": "FE reconstruction only -- skip transformer, EMA, masking, regression loss"},
    )
    recon_loss_type: str = field(
        default="l1",
        metadata={"help": "Reconstruction loss function: 'l1' or 'l2'"},
    )
    lambda_recon_fe: float = field(
        default=0.0,
        metadata={"help": "Weight for recon from pooled FE (post layer_norm) to input spectrogram"},
    )
    lambda_recon_trans: float = field(
        default=0.0,
        metadata={"help": "Weight for L1 recon from pooled transformer output to input spectrogram"},
    )
    recon_output_dim: int = field(
        default=245,
        metadata={"help": "Target spectrogram dimension for reconstruction heads"},
    )
    recon_decoder_hidden: str = field(
        default="512",
        metadata={"help": "Comma-separated hidden dims for recon MLP, or empty string for linear"},
    )
    recon_decoder_type: str = field(
        default="mlp",
        metadata={
            "help": "Decoder architecture for FE reconstruction: "
                    "'mlp' (mean-pool → MLP, default), "
                    "'linear' (mean-pool → Linear(512→245), no hidden layers), "
                    "'conv1d' (ConvTranspose1d upsampling, no mean-pool), "
                    "'interp' (interpolate to target len → per-step linear, no mean-pool), "
                    "'flat' (flatten all timesteps → MLP, no mean-pool), "
                    "'mirror' (ConvTranspose1d MirrorDecoder; sample-aligned upsampling; "
                    "needs_full_sequence=True; mirrors train_reconstruction.py heads)"
        },
    )

    # --- Per-component init (this PR; ``recon_components.py`` loaders) ---
    init_fe_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "Per-component override: conv feature_extractor source ckpt path."},
    )
    init_ln_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "Per-component override: post-FE LayerNorm (512-d) source ckpt path."},
    )
    init_proj_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "Per-component override: post_extract_proj source ckpt path."},
    )
    init_transformer_ckpt: Optional[str] = field(
        default=None,
        metadata={
            "help": "Per-component override: TransformerEncoder source ckpt path. "
                    "Supports fairseq ``encoder.*`` and data2vec_multi ``blocks.*`` (ViT QKV remap)."
        },
    )
    init_fe_recon_decoder_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "Per-component override: fe_recon_decoder source ckpt path "
                          "(Apr-28 FE-AE decoder key or train_reconstruction.py fe_mirror.*)."},
    )
    init_trans_recon_decoder_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "Per-component override: trans_recon_decoder source ckpt path "
                          "(train_reconstruction.py transformer_mirror.decoder.*)."},
    )

    # --- Per-component freezing (this PR) ---
    freeze_fe_v2: bool = field(
        default=False,
        metadata={"help": "Freeze conv feature_extractor only (also sets feature_grad_mult=0)."},
    )
    freeze_ln: bool = field(
        default=False, metadata={"help": "Freeze post-FE LayerNorm (512-d)."},
    )
    freeze_proj: bool = field(
        default=False, metadata={"help": "Freeze post_extract_proj."},
    )
    freeze_transformer_v2: bool = field(
        default=False, metadata={"help": "Freeze TransformerEncoder (alternative to existing freeze_encoder)."},
    )
    freeze_fe_recon_decoder: bool = field(
        default=False, metadata={"help": "Freeze fe_recon_decoder head."},
    )
    freeze_trans_recon_decoder: bool = field(
        default=False, metadata={"help": "Freeze trans_recon_decoder head."},
    )
    freeze_mask_emb: bool = field(
        default=False, metadata={"help": "Freeze mask embedding parameter (mask_emb)."},
    )
    freeze_final_proj: bool = field(
        default=False, metadata={"help": "Freeze final_proj (student head for EMA regression)."},
    )

    # --- Param-group tagging for composite optimizer (per-component LR) ---
    tag_param_groups: bool = field(
        default=False,
        metadata={
            "help": "If True, set ``p.param_group`` on every parameter to its natural component name "
                    "({fe, ln, proj, transformer, other}; fe_recon_decoder joins 'fe', "
                    "trans_recon_decoder joins 'transformer'). Required for ``optimizer._name: composite`` "
                    "with per-group LR; fully no-op when using a single Adam optimizer."
        },
    )

    # --- Cosine similarity maps (optional; rank 0; see cosim_epoch_utils + HOW_TO_RUN) ---
    epoch_cosim_enable: bool = field(
        default=False,
        metadata={"help": "If True, run cosine heatmaps on a fixed subset (by step or end-of-epoch)"},
    )
    epoch_cosim_interval_updates: int = field(
        default=1000,
        metadata={
            "help": "If >0, run cosine maps every N optimizer steps (skips end-of-epoch cosim). "
            "If 0, run only in task.end_epoch once per training epoch."
        },
    )
    epoch_cosim_subset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to .npy/.txt of train indices or structured_similarity JSON; see cosim_epoch_utils"
        },
    )
    epoch_cosim_structured_entries_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "If set, JSON with {\"entries\": [...]} — full structured panel (~100 samples across "
            "single_channel_all, multi_channel, sampled_data, labeled_data). Overrides epoch_cosim_subset_path. "
            "Generate with code/precompute_epoch_cosim_indices.py --structured_entries_json"
        },
    )
    epoch_cosim_max_samples: int = field(
        default=100,
        metadata={"help": "Max indices after loading subset"},
    )
    epoch_cosim_output_subdir: str = field(
        default="cosim_epoch",
        metadata={"help": "Subdir under checkpoint.save_dir for PNG heatmaps"},
    )
    epoch_cosim_micro_batch: int = field(
        default=8,
        metadata={"help": "Micro-batch size for epoch cosim forward passes"},
    )


def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


# ---------------------------------------------------------------------------
# Tier 1 ablation helpers
# ---------------------------------------------------------------------------

class IdentityFEProjection(nn.Module):
    """Drop-in replacement for ConvFeatureExtractionModel used in ablations.

    Maps each raw time step through a single linear layer without any striding,
    so the sequence length T is preserved (unlike the CNN FE which reduces T by
    the stride product).  Output shape matches ConvFE: [B, extractor_embed, T].
    """

    def __init__(self, extractor_embed: int):
        super().__init__()
        self.proj = nn.Linear(1, extractor_embed)

    def forward(self, x):  # x: [B, T]
        x = x.unsqueeze(-1)       # [B, T, 1]
        x = self.proj(x)          # [B, T, extractor_embed]
        return x.transpose(1, 2)  # [B, extractor_embed, T]


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg variance term: hinge loss pushing per-dim std above *gamma*.

    Args:
        z: embeddings of shape [N, D] or [B, T, D] (flattened internally).
        gamma: target standard deviation (default 1.0).
        eps: numerical stability offset added before sqrt.
    Returns:
        Scalar loss; 0 when every dimension already has std >= gamma.
    """
    if z.dim() == 3:
        z = z.reshape(-1, z.shape[-1])
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(gamma - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance term: penalise off-diagonal entries of the cov matrix.

    Args:
        z: embeddings of shape [N, D] or [B, T, D] (flattened internally).
    Returns:
        Scalar loss; 0 when embedding dimensions are fully decorrelated.
    """
    if z.dim() == 3:
        z = z.reshape(-1, z.shape[-1])
    N, D = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (N - 1)
    off_diag = cov.pow(2).sum() - cov.diag().pow(2).sum()
    return off_diag / D


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """Wang & Isola (ICML 2020) uniformity loss on the unit hypersphere.

    Args:
        z: embeddings of shape [N, D] (will be L2-normalised internally).
           For efficiency, pass at most ~1024 vectors.
        t: temperature (default 2.0).
    Returns:
        Scalar log-mean-exp loss; lower means more uniformly distributed.
    """
    if z.dim() == 3:
        z = z.reshape(-1, z.shape[-1])
    z = F.normalize(z, dim=1)
    sq_pdist = torch.pdist(z, p=2).pow(2)
    return torch.log(torch.exp(-t * sq_pdist).mean() + 1e-8)


class ReconstructionDecoder(nn.Module):
    """MLP (or linear) mapping pooled embeddings → spectrogram bins."""

    def __init__(self, in_dim: int, out_dim: int, hidden_spec: str):
        super().__init__()
        spec = (hidden_spec or "").strip()
        if not spec:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            dims = [int(x.strip()) for x in spec.split(",") if x.strip()]
            layers = []
            prev = in_dim
            for h in dims:
                layers.extend([nn.Linear(prev, h), nn.GELU()])
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            self.net = nn.Sequential(*layers)

    @property
    def needs_full_sequence(self) -> bool:
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Conv1dReconDecoder(nn.Module):
    """ConvTranspose1d decoder: [B, C, T'] → [B, 1, out_dim]. No mean-pool needed."""

    def __init__(self, in_channels: int, out_dim: int, fe_time_steps: int = 49):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(in_channels, 256, kernel_size=4, stride=2, padding=1),  # 49→98
            nn.GELU(),
            nn.ConvTranspose1d(256, 64, kernel_size=4, stride=2, padding=1),           # 98→196
            nn.GELU(),
            nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1),             # 196→392
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, fe_time_steps)
            conv_out_len = self.net(dummy).shape[-1]
        self.proj = nn.Linear(conv_out_len, out_dim)

    @property
    def needs_full_sequence(self) -> bool:
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T', C] (time-last from layer_norm) → [B, out_dim]"""
        x = x.transpose(1, 2)              # [B, C, T']
        x = self.net(x)                    # [B, 1, conv_out_len]
        x = x.squeeze(1)                   # [B, conv_out_len]
        return self.proj(x)                # [B, out_dim]


class InterpReconDecoder(nn.Module):
    """Interpolate FE sequence to target length, then per-step linear → scalar. No mean-pool."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    @property
    def needs_full_sequence(self) -> bool:
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T', C] → [B, out_dim]"""
        x = x.transpose(1, 2)                                # [B, C, T']
        x = F.interpolate(x, size=self.out_dim, mode="linear", align_corners=False)
        x = x.transpose(1, 2)                                # [B, out_dim, C]
        return self.fc(x).squeeze(-1)                         # [B, out_dim]


class MirrorReconDecoder(nn.Module):
    """ConvTranspose1d 5-layer decoder mirroring the SpectralFM FE in reverse.

    Same architecture as :class:`code.train_reconstruction.MirrorDecoder`. Operates on
    the **full FE sequence** (``needs_full_sequence=True``) → ``[B, 245]`` output,
    so it gives sample-aligned reconstructions instead of the data2vec MLP's
    mean-pooled-from-pooled-features output.

    Designed for ``conv_feature_layers='[(512, 3, 1)×4, (512, 5, 5)]'`` (245 → 47).
    """

    needs_full_sequence = True

    def __init__(self, in_channels: int = 512, out_dim: int = 245):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(self.in_channels, 512, kernel_size=5, stride=5, output_padding=2),
                nn.LayerNorm(237),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1),
                nn.LayerNorm(239),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1),
                nn.LayerNorm(241),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1),
                nn.LayerNorm(243),
                nn.GELU(),
            ),
            nn.ConvTranspose1d(512, 1, kernel_size=3, stride=1),
        ])

    def forward(self, fe_seq):
        """``fe_seq``: ``[B, T, C]`` (post-LN FE features) → ``[B, out_dim]``."""
        x = fe_seq.transpose(1, 2).contiguous()  # [B, C, T]
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(1)  # [B, out_dim]


class FlatReconDecoder(nn.Module):
    """Flatten all timesteps then MLP: [B, T'*C] → [B, out_dim]. No mean-pool."""

    def __init__(self, in_dim: int, out_dim: int, fe_time_steps: int = 49):
        super().__init__()
        flat_dim = in_dim * fe_time_steps  # 512*49 = 25088
        bottleneck = 512
        self.net = nn.Sequential(
            nn.Linear(flat_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, out_dim),
        )

    @property
    def needs_full_sequence(self) -> bool:
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T', C] → [B, out_dim]"""
        return self.net(x.reshape(x.shape[0], -1))


@register_model("data2vec_audio", dataclass=Data2VecAudioConfig)
class Data2VecAudioModel(BaseFairseqModel):
    def __init__(self, cfg: Data2VecAudioConfig):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.extractor_embed = feature_enc_layers[-1][0]

        self.model_path = cfg.model_path
        if self.model_path:
            print(f"model_path = { self.model_path }")
        self.ema = None
        self.embed = cfg.encoder_embed_dim

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        # --- Tier 1: FE mode wiring ---
        fe_mode = getattr(cfg, "fe_mode", "train")
        if fe_mode == "identity":
            self.feature_extractor = IdentityFEProjection(self.extractor_embed)
            logger.info("[Tier1] FE replaced with IdentityFEProjection (no striding)")
        elif fe_mode == "frozen":
            # feature_grad_mult=0 routes FE forward through torch.no_grad(),
            # so no gradients reach the optimizer for FE parameters.
            self.feature_grad_mult = 0.0
            logger.info("[Tier1] FE frozen (feature_grad_mult=0, weights from cfg.model_path)")

        self.post_extract_proj = build_post_extract_proj(
            self.extractor_embed,
            cfg.encoder_embed_dim,
            getattr(cfg, "post_extract_proj_type", "linear"),
            int(getattr(cfg, "post_extract_proj_mlp_hidden", 1536)),
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space
        self.mask_seed = cfg.mask_seed  # For deterministic masking

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.extractor_embed)

        # --- Tier 1: encoder freezing ---
        if getattr(cfg, "freeze_encoder", False):
            for p in self.encoder.parameters():
                p.requires_grad = False
            logger.info("[Tier1] Transformer encoder frozen")

        self.final_proj = nn.Linear(self.embed, self.embed)

        hspec = getattr(cfg, "recon_decoder_hidden", "512") or "512"
        out_d = int(getattr(cfg, "recon_output_dim", 245))
        dec_type = getattr(cfg, "recon_decoder_type", "mlp")
        fe_time_steps = 49  # fixed for conv_feature_layers with 245-sample input

        if dec_type == "conv1d":
            self.fe_recon_decoder = Conv1dReconDecoder(self.extractor_embed, out_d, fe_time_steps)
        elif dec_type == "interp":
            self.fe_recon_decoder = InterpReconDecoder(self.extractor_embed, out_d)
        elif dec_type == "flat":
            self.fe_recon_decoder = FlatReconDecoder(self.extractor_embed, out_d, fe_time_steps)
        elif dec_type == "linear":
            self.fe_recon_decoder = ReconstructionDecoder(self.extractor_embed, out_d, "")
        elif dec_type == "mirror":
            self.fe_recon_decoder = MirrorReconDecoder(self.extractor_embed, out_d)
        else:
            self.fe_recon_decoder = ReconstructionDecoder(self.extractor_embed, out_d, hspec)
        logger.info(f"[recon] FE decoder type={dec_type}, class={type(self.fe_recon_decoder).__name__}")

        # trans_recon_decoder: 'mirror' upsamples from transformer output (768→512 stem first).
        if dec_type == "mirror":
            # Wrap a stem 1×1 conv (embed→512) so transformer features feed the same
            # MirrorReconDecoder used for the FE branch, with sample-aligned output.
            class _TransMirrorWrap(nn.Module):
                needs_full_sequence = True
                def __init__(self, in_dim, mid_dim, out_dim):
                    super().__init__()
                    self.stem = nn.Conv1d(in_dim, mid_dim, kernel_size=1)
                    self.pre_ln = nn.LayerNorm(mid_dim)
                    self.body = MirrorReconDecoder(mid_dim, out_dim)
                def forward(self, x_btc):
                    x = x_btc.transpose(1, 2).contiguous()
                    x = self.stem(x)
                    x = self.pre_ln(x.transpose(1, 2))
                    return self.body(x)
            self.trans_recon_decoder = _TransMirrorWrap(self.embed, 512, out_d)
        else:
            self.trans_recon_decoder = ReconstructionDecoder(self.embed, out_d, hspec)

        self.num_updates = 0
        
        # For sanity testing: fixed mask indices range
        self._fixed_mask_start = None
        self._fixed_mask_end = None
        
        # For loss validation: save masks during training
        self._mask_save_dir = getattr(cfg, 'mask_save_dir', None)
        if self._mask_save_dir:
            import os
            os.makedirs(self._mask_save_dir, exist_ok=True)
            print(f"[+] Mask saving enabled. Saving to: {self._mask_save_dir}")
        
        # For fixed masking memory: store masks keyed by batch index
        # Note: masks are computed on embeddings per batch, so we use batch index as key
        self._mask_memory: Dict[int, torch.Tensor] = {}
        self._use_mask_memory: bool = False
        self._mask_memory_counter: int = 0  # Global counter for unique keys across batches

        # Enable debug plots only when debug mode is explicitly enabled
        # Check for debug flag in config (can be set via --debug or model.debug=True)
        debug_enabled = getattr(cfg, 'debug', False)
        if debug_enabled:
            set_debug_plot_mode('train')

        if getattr(cfg, "recon_only", False):
            self.final_proj = None
            for name, p in self.named_parameters():
                if "feature_extractor" not in name and "fe_recon_decoder" not in name:
                    p.requires_grad = False
            logger.info("[recon_only] Only FE + fe_recon_decoder trainable; "
                        "transformer/EMA/regression skipped in forward()")
        elif cfg.train_only_fe and getattr(cfg, "fe_mode", "train") == "train":
            self.freeze_all_except_feature_extractor()
            if getattr(cfg, "lambda_recon_fe", 0.0) > 0:
                for p in self.fe_recon_decoder.parameters():
                    p.requires_grad = True
            if getattr(cfg, "lambda_recon_trans", 0.0) > 0:
                for p in self.trans_recon_decoder.parameters():
                    p.requires_grad = True
        elif not cfg.train_only_fe:
            print("[+] Training the entire model (not only Feature-Extractor)")
            logger.info("[+] Training the entire model (not only Feature-Extractor)")
            if getattr(cfg, "lambda_recon_fe", 0.0) > 0:
                for p in self.fe_recon_decoder.parameters():
                    p.requires_grad = True
            if getattr(cfg, "lambda_recon_trans", 0.0) > 0:
                for p in self.trans_recon_decoder.parameters():
                    p.requires_grad = True

    def freeze_all_except_feature_extractor(self):
        for name, p in self.named_parameters():
            if "feature_extractor" not in name:
                p.requires_grad = False

        print("[INFO] ONLY feature extractor is trainable")
        logger.info("[INFO] ONLY feature extractor is trainable")

    def enable_mask_memory(self):
        """Enable mask memory mode - masks will be stored and reused."""
        self._use_mask_memory = True
        logger.info("[+] Mask memory enabled")

    def disable_mask_memory(self):
        """Disable mask memory mode - masks will be generated normally."""
        self._use_mask_memory = False
        logger.info("[+] Mask memory disabled")

    def clear_mask_memory(self):
        """Clear all stored masks from memory."""
        self._mask_memory.clear()
        self._mask_memory_counter = 0
        logger.info("[+] Mask memory cleared")

    def save_mask_memory(self, path: str):
        """Save mask memory to disk."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        # Convert tensors to CPU and numpy for serialization
        mask_memory_np = {
            sample_id: mask.cpu().numpy() 
            for sample_id, mask in self._mask_memory.items()
        }
        torch.save(mask_memory_np, path)
        logger.info(f"[+] Saved {len(self._mask_memory)} masks to {path}")

    def load_mask_memory(self, path: str):
        """Load mask memory from disk."""
        import os
        if not os.path.exists(path):
            logger.warning(f"[!] Mask memory file not found: {path}")
            return False
        mask_memory_np = torch.load(path, map_location='cpu')
        # Convert numpy arrays back to tensors
        self._mask_memory = {
            sample_id: torch.from_numpy(mask) 
            for sample_id, mask in mask_memory_np.items()
        }
        logger.info(f"[+] Loaded {len(self._mask_memory)} masks from {path}")
        return True

    def load_model_weights(self, state, model, cfg):
        if "_ema" in state["model"]:
            del state["model"]["_ema"]
        model.load_state_dict(state["model"], strict=True)
        print("Loaded model weights")

    def make_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,
        )
        skip_keys = set()
        if self.cfg.ema_layers_only:
            self.cfg.ema_transformer_only = True
            for k, _ in self.encoder.pos_conv.named_parameters():
                skip_keys.add(f"pos_conv.{k}")

        self.ema = EMAModule(
            self.encoder if self.cfg.ema_transformer_only else self,
            ema_config,
            skip_keys=skip_keys,
        )

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is None and self.final_proj is not None and not getattr(self.cfg, "recon_only", False):
            logger.info(f"making ema teacher")
            self.make_ema_teacher()
        elif self.training and self.ema is not None:
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.encoder if self.cfg.ema_transformer_only else self)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.ema is not None:
            k = prefix + "_ema"
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg: Data2VecAudioConfig, task=None):
        """Build a new model instance."""
        model = cls(cfg)
        if cfg.model_path and not cfg.skip_pretrained_weights:
            logger.info(f"Loading pretrained checkpoint from {cfg.model_path}")
            print(f"Loading pretrained checkpoint from {cfg.model_path}")
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.model_path, {})
            ckpt = state.get("model", state)
            ckpt.pop("_ema", None)

            fe_mode = getattr(cfg, "fe_mode", "train")
            if fe_mode == "frozen":
                # Load only the feature_extractor weights from the pretrained checkpoint.
                # This lets us initialise the FE with Librispeech-pretrained weights
                # while keeping the transformer randomly initialised.
                fe_weights = {
                    k[len("feature_extractor."):]: v
                    for k, v in ckpt.items()
                    if k.startswith("feature_extractor.")
                }
                model.feature_extractor.load_state_dict(fe_weights, strict=True)
                logger.info(
                    f"[Tier1] Loaded pretrained FE weights ({len(fe_weights)} tensors) "
                    f"from {cfg.model_path}"
                )
            else:
                model.load_state_dict(ckpt, strict=False)
                logger.info(f"Loaded full pretrained checkpoint (strict=False)")

        # ── Per-component init / freeze / param-group tagging (this PR) ──
        _maybe_apply_recon_components(model, cfg)
        return model

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
        sample_indices=None,
    ):
        """
        Apply masking to features.
        
        Args:
            x: features tensor of shape (B, T, C)
            padding_mask: padding mask
            mask_indices: precomputed mask indices (optional)
            mask_channel_indices: precomputed channel mask indices (optional)
            sample_indices: tensor of sample IDs (optional).
                           - If mask memory is enabled: used to load/store masks from memory
                           - If mask_seed is set: used for deterministic masking (and saved to memory if enabled)
                           - Otherwise: used for mask memory storage if enabled
        """
        B, T, C = x.shape

        # Prepare deterministic masking parameters if configured
        # When mask_seed is set, use epoch=0 (fixed) so that each sample always gets
        # the same mask regardless of when it's processed.
        # This ensures: mask = hash(seed, 0, sample_id) is consistent across train/eval.
        # If mask memory is also enabled, these deterministic masks will be saved to memory.
        seed = None
        epoch = None
        indices = None
        if self.mask_seed is not None and sample_indices is not None:
            seed = self.mask_seed
            epoch = 0  # Fixed epoch ensures same sample always gets same mask
            indices = sample_indices

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
                seed=seed,
                epoch=epoch,
                indices=indices,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            if mask_indices is None:
                # Check mask memory first if enabled and sample_indices are provided
                use_memory_masks = (
                    self._use_mask_memory 
                    and sample_indices is not None 
                    and not self.training  # Only use memory in eval mode
                )
                
                if use_memory_masks:
                    # Try to load masks from memory
                    # Note: mask_memory stores masks keyed by global counter (0, 1, 2, ...)
                    # Masks are computed on embeddings per batch, stored sequentially
                    # During evaluation, we need to reset the counter and load sequentially
                    mask_indices_list = []
                    all_masks_found = True
                    num_samples = len(sample_indices)
                    
                    # Reset counter at start of evaluation if needed
                    if not hasattr(self, '_mask_memory_eval_counter'):
                        self._mask_memory_eval_counter = 0
                    
                    for i in range(num_samples):
                        # Use global counter to look up mask sequentially
                        key = self._mask_memory_eval_counter
                        if key in self._mask_memory:
                            stored_mask = self._mask_memory[key]
                            # Handle sequence length differences
                            if stored_mask.shape[0] == T:
                                mask_indices_list.append(stored_mask)
                                self._mask_memory_eval_counter += 1
                            else:
                                # Sequence length mismatch - fall back to generation
                                logger.warning(
                                    f"Sequence length mismatch for mask key {key}: "
                                    f"stored={stored_mask.shape[0]}, current={T}. "
                                    f"Falling back to random generation."
                                )
                                all_masks_found = False
                                break
                        else:
                            # Mask not found in memory - fall back to generation
                            logger.debug(
                                f"Mask key {key} not found in mask_memory. "
                                f"Available keys: {sorted(list(self._mask_memory.keys()))[:10]}. "
                                f"Falling back to random generation."
                            )
                            all_masks_found = False
                            break
                    
                    if all_masks_found and len(mask_indices_list) == num_samples:
                        # All masks found in memory - stack them
                        mask_indices = torch.stack(mask_indices_list).to(x.device)
                    else:
                        # Fall back to normal generation
                        use_memory_masks = False
                
                if not use_memory_masks:
                    # Generate masks: use seed-based if mask_seed is set, otherwise use random masking
                    # Note: seed/epoch/indices are already set above based on mask_seed
                    mask_indices = compute_mask_indices(
                        (B, T),
                        padding_mask,
                        self.mask_prob,
                        self.mask_length,
                        self.mask_selection,
                        self.mask_other,
                        min_masks=1,
                        no_overlap=self.no_mask_overlap,
                        min_space=self.mask_min_space,
                        require_same_masks=self.cfg.require_same_masks,
                        mask_dropout=self.cfg.mask_dropout,
                        seed=seed,  # None if mask_seed not set, otherwise uses deterministic masking
                        epoch=epoch,  # None if mask_seed not set, otherwise uses deterministic masking
                        indices=indices,  # None if mask_seed not set, otherwise uses deterministic masking
                    )
                    mask_indices = torch.from_numpy(mask_indices).to(x.device)
                    
                    # Store masks in memory if memory is enabled and we're in eval mode
                    # This saves both deterministic (if mask_seed set) and random masks
                    # Note: We don't require sample_indices for storage since we use a global counter
                    if (
                        self._use_mask_memory 
                        and not self.training
                    ):
                        # mask_indices shape is (B_mask, T) where B_mask is the batch size of the embeddings
                        # Use the actual batch dimension from mask_indices, not the input batch size B,
                        # since masks are computed on embeddings which may have different batch structure
                        # Store masks keyed by global counter to ensure unique keys across batches
                        B_mask = mask_indices.shape[0]
                        for i in range(B_mask):
                            # Store mask on CPU to save memory, keyed by global counter
                            # If mask_seed was set, this stores deterministic masks; otherwise random masks
                            self._mask_memory[self._mask_memory_counter] = mask_indices[i].cpu().clone()
                            self._mask_memory_counter += 1
                        # Debug: log first few masks being stored
                        if len(self._mask_memory) <= 5 or len(self._mask_memory) % 100 == 0:
                            logger.info(f"[DEBUG] Stored {len(self._mask_memory)} masks in memory (latest key: {self._mask_memory_counter - 1})")
                
                # Save masks if mask_save_dir is set (for loss validation)
                if hasattr(self, '_mask_save_dir') and self._mask_save_dir is not None:
                    import numpy as np
                    import os
                    if not hasattr(self, '_mask_batch_counter'):
                        self._mask_batch_counter = 0
                    mask_path = os.path.join(self._mask_save_dir, f"mask_batch_{self._mask_batch_counter:06d}.npy")
                    np.save(mask_path, mask_indices.cpu().numpy())
                    self._mask_batch_counter += 1
            
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                    seed=seed,
                    epoch=epoch,
                    indices=indices,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
                )
            x = index_put(x, mask_channel_indices, 0)

        return x, mask_indices

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def forward(
        self,
        source,
        padding_mask=None,
        mask=True,
        features_only=False,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
        sample_indices=None,
    ):
        features = source

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(features)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(features)

        features = features.transpose(1, 2)

        features = self.layer_norm(features)
        fe_seq = features  # pre-mask FE for reconstruction loss

        # --- recon_only: early return, skip transformer/EMA/masking ---
        if getattr(self.cfg, "recon_only", False):
            result = {"losses": {}}
            recon_target = self._recon_target(source)
            if getattr(self.fe_recon_decoder, "needs_full_sequence", False):
                pred_fe = self.fe_recon_decoder(fe_seq)
            else:
                pred_fe = self.fe_recon_decoder(self._mean_pool(fe_seq, padding_mask))
            loss_fn = F.mse_loss if getattr(self.cfg, "recon_loss_type", "l1") == "l2" else F.l1_loss
            result["losses"]["recon_fe"] = (
                loss_fn(pred_fe, recon_target, reduction="mean")
                * self.cfg.lambda_recon_fe
            )
            return result

        orig_padding_mask = padding_mask

        if padding_mask is not None and padding_mask.any():
            if getattr(self.cfg, "fe_mode", "train") == "identity":
                # IdentityFEProjection preserves sequence length T, so the
                # padding_mask from the dataloader (shape [B, T]) is already
                # correct — no conv-stride recalculation needed.
                pass
            else:
                input_lengths = (1 - padding_mask.long()).sum(-1)
                # apply conv formula to get real output_lengths
                output_lengths = self._get_feat_extract_output_lengths(input_lengths)

                padding_mask = torch.zeros(
                    features.shape[:2], dtype=features.dtype, device=features.device
                )

                # these two operations makes sure that all values
                # before the output lengths indices are attended to
                padding_mask[
                    (
                        torch.arange(padding_mask.shape[0], device=padding_mask.device),
                        output_lengths - 1,
                    )
                ] = 1
                padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        post_extract_stats = {}
        if self.post_extract_proj is not None:
            pe_in = features.detach().float()
            features = self.post_extract_proj(features)
            pe_out = features.detach().float()
            post_extract_stats = {
                "post_extract_in_mean": pe_in.mean(),
                "post_extract_in_std": pe_in.std(),
                "post_extract_out_mean": pe_out.mean(),
                "post_extract_out_std": pe_out.std(),
            }

        pre_encoder_features = None
        if self.cfg.ema_transformer_only:
            pre_encoder_features = features.clone()

        features = self.dropout_input(features)

        if mask:
            features_before_mask = features.clone()  # Store for debug plotting
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
                sample_indices=sample_indices,
            )
            # Debug plotting (controlled by config flag)
            if getattr(self.cfg, 'debug_plot_masking', False):
                debug_plot_input_masking(source, features_before_mask, mask_indices, x, sample_indices=sample_indices)
        else:
            x = features
            mask_indices = None

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=layer,
        )

        if features_only:
            out = {
                "x": x,
                "padding_mask": padding_mask,
                "layer_results": layer_results,
            }
            out.update(post_extract_stats)
            return out

        result = {
            "losses": {},
        }
        result.update(post_extract_stats)

        # Check if EMA is initialized (fallback for edge cases)
        # EMA should be initialized via set_num_updates() when loading checkpoints.
        # However, if optimizer_history is missing or num_updates is not available,
        # EMA might not be initialized. This fallback handles those cases.
        if self.ema is None: 
            if self.final_proj is None:
                # Model has been converted for fine-tuning (remove_pretraining_modules called)
                # EMA is not needed in this case, but forward() should not be called with mask=True
                raise RuntimeError(
                    "EMA is None and final_proj is None. "
                    "This model has been converted for fine-tuning and cannot compute pretraining loss. "
                    "Use extract_features() with features_only=True instead."
                )
            # Fallback: Initialize EMA lazily if somehow not initialized
            # This can happen if checkpoint doesn't have optimizer_history or num_updates
            logger.warning(
                "EMA not initialized during model loading - initializing lazily from current model state. "
                "This may indicate the checkpoint is missing optimizer_history/num_updates."
            )
            self.make_ema_teacher()
            if self.cfg.ema_transformer_only:
                self.ema.model.load_state_dict(self.encoder.state_dict())
            else:
                self.ema.model.load_state_dict(self.state_dict())

        with torch.no_grad():
            self.ema.model.eval()

            # Get the dtype of the EMA model to match inputs
            # This handles cases where EMA model might be in fp16 or fp32
            ema_dtype = next(self.ema.model.parameters()).dtype
            
            if self.cfg.ema_transformer_only:
                # Convert pre_encoder_features to match EMA model dtype
                if pre_encoder_features.dtype != ema_dtype:
                    pre_encoder_features_ema = pre_encoder_features.to(dtype=ema_dtype)
                else:
                    pre_encoder_features_ema = pre_encoder_features
                y, layer_results = self.ema.model.extract_features(
                    pre_encoder_features_ema,
                    padding_mask=padding_mask,
                    min_layer=self.cfg.encoder_layers - self.average_top_k_layers,
                )
                y = {
                    "x": y,
                    "padding_mask": padding_mask,
                    "layer_results": layer_results,
                }
            else:
                # Convert source to match EMA model dtype
                if source.dtype != ema_dtype:
                    source_ema = source.to(dtype=ema_dtype)
                else:
                    source_ema = source
                y = self.ema.model.extract_features(
                    source=source_ema,
                    padding_mask=orig_padding_mask,
                    mask=False,
                )

            target_layer_results = [l[2] for l in y["layer_results"]]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    tl.permute(1, 2, 0) for tl in target_layer_results  # TBC -> BCT
                ]
                permuted = True

            if self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]

            if self.cfg.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]

            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]

            if self.cfg.group_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-2:])
                    for tl in target_layer_results
                ]

            if self.cfg.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]

            y = sum(target_layer_results) / len(target_layer_results)

            if self.cfg.layer_norm_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

            if self.cfg.instance_norm_targets:
                y = F.instance_norm(y.float().transpose(1, 2)).transpose(1, 2)

            if not permuted:
                y = y.transpose(0, 1)

            y = y[mask_indices]

        # Stash full encoder output (all positions) for regularization losses.
        # detach() so that reg-loss gradients flow only through their own path
        # and do not interfere with the EMA target computation.
        x_for_reg = x.reshape(-1, x.shape[-1]).detach()

        recon_target = self._recon_target(source)
        recon_loss_fn = F.mse_loss if getattr(self.cfg, "recon_loss_type", "l1") == "l2" else F.l1_loss
        if getattr(self.cfg, "lambda_recon_fe", 0.0) > 0:
            if getattr(self.fe_recon_decoder, "needs_full_sequence", False):
                pred_fe = self.fe_recon_decoder(fe_seq)
            else:
                pred_fe = self.fe_recon_decoder(self._mean_pool(fe_seq, padding_mask))
            result["losses"]["recon_fe"] = (
                recon_loss_fn(pred_fe, recon_target, reduction="mean") * self.cfg.lambda_recon_fe
            )
        if getattr(self.cfg, "lambda_recon_trans", 0.0) > 0:
            if getattr(self.trans_recon_decoder, "needs_full_sequence", False):
                pred_t = self.trans_recon_decoder(x)
            else:
                tp = self._mean_pool(x, padding_mask)
                pred_t = self.trans_recon_decoder(tp)
            result["losses"]["recon_trans"] = (
                recon_loss_fn(pred_t, recon_target, reduction="mean") * self.cfg.lambda_recon_trans
            )

        x = x[mask_indices]
        x = self.final_proj(x)

        sz = x.size(-1)

        # Debug plotting (controlled by config flag)
        if getattr(self.cfg, 'debug_plot_predictions', False):
            debug_plot_predictions(x, y)

        if self.loss_beta == 0: # fixme noy debug 
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                x.float(), y.float(), reduction="none", beta=self.loss_beta
            ).sum(dim=-1)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(sz)

        result["losses"]["regression"] = loss.sum() * scale

        # --- Tier 1 regularization losses ---
        if getattr(self.cfg, "lambda_var", 0.0) > 0:
            vl = variance_loss(x_for_reg, getattr(self.cfg, "var_gamma", 1.0))
            result["losses"]["var"] = vl * self.cfg.lambda_var
            result["var"] = vl.detach()
        if getattr(self.cfg, "lambda_cov", 0.0) > 0:
            cl = covariance_loss(x_for_reg)
            result["losses"]["cov"] = cl * self.cfg.lambda_cov
            result["cov"] = cl.detach()
        if getattr(self.cfg, "lambda_uniform", 0.0) > 0:
            # uniformity is O(N²); subsample to at most 1024 vectors for efficiency
            emb_u = x_for_reg
            if emb_u.shape[0] > 1024:
                idx = torch.randperm(emb_u.shape[0], device=emb_u.device)[:1024]
                emb_u = emb_u[idx]
            ul = uniformity_loss(emb_u)
            result["losses"]["uniform"] = ul * self.cfg.lambda_uniform
            result["uniform"] = ul.detach()

        if "sample_size" not in result:
            result["sample_size"] = loss.numel()

        with torch.no_grad():
            result["target_var"] = self.compute_var(y)
            result["pred_var"] = self.compute_var(x.float())

        if self.num_updates > 5000 and result["target_var"] < self.cfg.min_target_var:
            logger.error(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
            raise Exception(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
        if self.num_updates > 5000 and result["pred_var"] < self.cfg.min_pred_var:
            logger.error(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )
            raise Exception(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )

        if self.ema is not None:
            result["ema_decay"] = self.ema.get_decay() * 1000

        return result

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    def _mean_pool(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mean-pool over time; padding_mask True = padded (ignored)."""
        if padding_mask is None:
            return x.mean(dim=1)
        m = (~padding_mask).float().unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1e-8)
        return (x * m).sum(dim=1) / denom

    def _recon_target(self, source: torch.Tensor) -> torch.Tensor:
        d = int(getattr(self.cfg, "recon_output_dim", 245))
        x = source.float()
        if x.dim() == 2:
            if x.shape[1] >= d:
                return x[:, :d]
            return F.pad(x, (0, d - x.shape[1]))
        flat = x.reshape(x.shape[0], -1)
        if flat.shape[1] >= d:
            return flat[:, :d]
        return F.pad(flat, (0, d - flat.shape[1]))

    def extract_cosim_epoch_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input / pooled FE (post layer_norm) / pooled transformer emb for epoch cosine maps."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                features = source
                if self.feature_grad_mult > 0:
                    features = self.feature_extractor(features)
                else:
                    with torch.no_grad():
                        features = self.feature_extractor(features)
                features = features.transpose(1, 2)
                features = self.layer_norm(features)
                fe_seq = features
                if padding_mask is not None and padding_mask.any():
                    if getattr(self.cfg, "fe_mode", "train") == "identity":
                        pass
                    else:
                        input_lengths = (1 - padding_mask.long()).sum(-1)
                        output_lengths = self._get_feat_extract_output_lengths(input_lengths)
                        padding_mask = torch.zeros(
                            features.shape[:2], dtype=features.dtype, device=features.device
                        )
                        padding_mask[
                            (
                                torch.arange(padding_mask.shape[0], device=padding_mask.device),
                                output_lengths - 1,
                            )
                        ] = 1
                        padding_mask = (
                            1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])
                        ).bool()
                else:
                    padding_mask = None

                fe_pool = self._mean_pool(fe_seq, padding_mask)
                if self.post_extract_proj is not None:
                    features = self.post_extract_proj(features)
                features = self.dropout_input(features)
                x, _ = self.encoder(features, padding_mask=padding_mask)
                emb_pool = self._mean_pool(x, padding_mask)
                inp = self._recon_target(source)
                return inp, fe_pool, emb_pool
        finally:
            if was_training:
                self.train()

    def extract_features(
        self, source, padding_mask, mask=False, layer=None
    ):
        res = self.forward(
            source,
            padding_mask,
            mask=mask,
            features_only=True,
            layer=layer,
        )
        return res

    def remove_pretraining_modules(self, last_layer=None):
        self.final_proj = None
        self.ema = None
        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )
