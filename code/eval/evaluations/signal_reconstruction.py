"""
Signal Reconstruction Evaluation (true reconstruction through the full pipeline)
---------------------------------------------------------------------------------
Ports the reconstruction flow from compare_fe_vs_trans_recon.py / train_reconstruction.py
(branch: reconstruction-loss-experiments) into the fairseq-free eval package.

Three reconstruction pathways, each loaded from its OWN checkpoint:

  FE recon         : input → FE → LayerNorm → MirrorDecoder → signal
                     checkpoint keys: encoder / layer_norm / decoder, OR a native
                     fairseq checkpoint carrying an fe_recon_decoder (e.g. Step 1)
                     (e.g. autoencoder_experiments/fe_signal_recon_*/ckpt_*.pt)

  Projection recon : input → FE → LN → proj → TransformerMirrorDecoder → signal
                     (stops before the transformer encoder)
                     checkpoint: native fairseq checkpoint carrying a
                     proj_recon_decoder (e.g. a Step 2-style run)

  Transformer recon: input → FE → LN → proj → Transformer → TransformerMirrorDecoder → signal
                     checkpoint keys: transformer_mirror / backbone_ckpt, OR a native
                     fairseq checkpoint carrying a trans_recon_decoder (e.g. Step 3)
                     The backbone (FE+LN+proj+transformer) is loaded from the fairseq
                     checkpoint referenced by `backbone_ckpt` (remapped into the HF
                     Data2VecAudioModel — no fairseq import needed).
                     (e.g. autoencoder_experiments/transformer_recon_*/ckpt_tr_*.pt)

Any combination can run — pass fe_ckpt / proj_ckpt / tr_ckpt independently.

Normalization: if `normalize` was recorded in the checkpoint (newer training runs,
--normalize flag), the same per-sample F.layer_norm (zero-mean unit-std) is applied
to the input; the (normalized) input is the reconstruction target. Override with
the `normalize` argument.

Score: per-sample MSE(reconstruction, target) for each pathway.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..model import FairseqConvFeatureExtractor, MirrorDecoder, TransformerMirrorDecoder
from ..signal_features import STRATIFIER_ORDER, compute_signal_features

_FE_LAYERS = [(512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 5, 5)]


def load_fe_recon(ckpt_path: str, device: str = "cpu"):
    """
    Load a standalone FE autoencoder checkpoint (keys: encoder / layer_norm / decoder).
    Returns (fe, ln, decoder, meta).
    """
    import torch.nn as nn

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("encoder", "layer_norm", "decoder"):
        if key not in sd:
            raise ValueError(f"Not an FE-recon checkpoint (missing '{key}'): {ckpt_path}")

    fe = FairseqConvFeatureExtractor(_FE_LAYERS)
    ln = nn.LayerNorm(512)
    decoder = MirrorDecoder()

    for name, mod, state in (("FE", fe, sd["encoder"]),
                             ("LN", ln, sd["layer_norm"]),
                             ("Decoder", decoder, sd["decoder"])):
        r = mod.load_state_dict(state, strict=False)
        if r.missing_keys or r.unexpected_keys:
            raise RuntimeError(
                f"FE-recon load error: {name} missing={r.missing_keys} unexpected={r.unexpected_keys}"
            )
        mod.eval().to(device)

    meta = {k: sd.get(k) for k in ("n_samples", "steps", "lr", "warmup", "tag", "normalize")}
    print(f"[SignalRecon] FE-recon loaded: {os.path.basename(ckpt_path)}  meta={meta}")
    return fe, ln, decoder, meta


def is_native_fe_recon_ckpt(ckpt_path: str) -> bool:
    """
    True if ckpt_path is a plain fairseq hydra_train checkpoint (keys: cfg/model,
    the format every step0/step1-style training run actually produces) with an
    fe_recon_decoder attached — as opposed to load_fe_recon's standalone-save
    format (top-level encoder/layer_norm/decoder keys) or load_3ae_recon's
    combined format (data2vec_audio/fe_mirror keys).
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return (
        "cfg" in sd and "model" in sd
        and any(k.startswith("fe_recon_decoder.") for k in sd["model"])
    )


def load_native_fe_recon(ckpt_path: str, device: str = "cpu"):
    """
    Load FE + fe_recon_decoder straight out of a native fairseq hydra_train
    checkpoint (e.g. a Step 1-style joint FE+decoder run) — the decoder lives
    inline as a submodule of the full model (`fe_recon_decoder.*` keys)
    instead of a separate standalone save. FairseqConvFeatureExtractor's keys
    already match the raw checkpoint 1:1 (see model.py), and the native
    MirrorReconDecoder (data2vec_audio.py) and this package's MirrorDecoder
    are architecturally identical layer-for-layer, so no remapping is needed
    beyond stripping the `fe_recon_decoder.` prefix.

    Returns (fe, ln, decoder, meta) — same shape as load_fe_recon, so it
    drops into run()'s existing downstream code unchanged.
    """
    import torch.nn as nn

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not ("cfg" in raw and "model" in raw):
        raise ValueError(f"Not a native fairseq checkpoint: {ckpt_path}")
    sd = raw["model"]
    if not any(k.startswith("fe_recon_decoder.") for k in sd):
        raise ValueError(f"No fe_recon_decoder found in checkpoint: {ckpt_path}")

    fe = FairseqConvFeatureExtractor(_FE_LAYERS)
    ln = nn.LayerNorm(512)
    decoder = MirrorDecoder()

    fe_state = {k[len("feature_extractor."):]: v for k, v in sd.items()
                if k.startswith("feature_extractor.")}
    ln_state = {k[len("layer_norm."):]: v for k, v in sd.items()
                if k in ("layer_norm.weight", "layer_norm.bias")}
    dec_state = {k[len("fe_recon_decoder."):]: v for k, v in sd.items()
                 if k.startswith("fe_recon_decoder.")}

    for name, mod, state in (("FE", fe, fe_state), ("LN", ln, ln_state), ("Decoder", decoder, dec_state)):
        r = mod.load_state_dict(state, strict=False)
        if r.missing_keys or r.unexpected_keys:
            raise RuntimeError(
                f"native FE-recon load error: {name} missing={r.missing_keys} unexpected={r.unexpected_keys}"
            )
        mod.eval().to(device)

    cfg = raw.get("cfg", {})
    model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else getattr(cfg, "model", cfg)
    lambda_recon_fe = (model_cfg.get("lambda_recon_fe") if isinstance(model_cfg, dict)
                        else getattr(model_cfg, "lambda_recon_fe", None))
    task_cfg = cfg.get("task", cfg) if isinstance(cfg, dict) else getattr(cfg, "task", cfg)
    normalize = (task_cfg.get("normalize") if isinstance(task_cfg, dict)
                 else getattr(task_cfg, "normalize", True))
    meta = {"lambda_recon_fe": lambda_recon_fe, "normalize": bool(normalize)}
    print(f"[SignalRecon] native FE-recon loaded: {os.path.basename(ckpt_path)}  meta={meta}")
    return fe, ln, decoder, meta


def is_native_proj_recon_ckpt(ckpt_path: str) -> bool:
    """
    True if ckpt_path is a plain fairseq hydra_train checkpoint (keys: cfg/model)
    with a proj_recon_decoder attached (a Step 2-style projection-reconstruction
    run) — the decoder lives inline as `proj_recon_decoder.*` keys alongside the
    full backbone, as opposed to load_tr_recon's separate transformer_mirror
    standalone-save format.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return (
        "cfg" in sd and "model" in sd
        and any(k.startswith("proj_recon_decoder.") for k in sd["model"])
    )


class _ProjMirrorHead(torch.nn.Module):
    """
    Reimplements data2vec_audio.py's `_TransMirrorWrap` (the class actually
    instantiated for proj_recon_decoder when recon_decoder_type='mirror'):
    a 1x1 conv stem (embed_dim→mid_dim) + LayerNorm(mid_dim) + a
    MirrorReconDecoder body — NOT the eval package's own TransformerMirrorDecoder
    (different key names: stem/pre_ln/body.layers vs proj/pre_decoder_ln/decoder.layers).
    `body` reuses this package's MirrorDecoder since MirrorReconDecoder's `layers.*`
    keys already match it 1:1 (same as load_native_fe_recon's fe_recon_decoder case);
    only the surrounding forward-pass shape convention (channels-last in, matching
    the native class's own transpose calls) differs and is reproduced here.
    """

    def __init__(self, in_dim: int = 768, mid_dim: int = 512, out_dim: int = 245):
        super().__init__()
        import torch.nn as nn
        self.stem = nn.Conv1d(in_dim, mid_dim, kernel_size=1)
        self.pre_ln = nn.LayerNorm(mid_dim)
        self.body = MirrorDecoder()

    def forward(self, x_btc: torch.Tensor) -> torch.Tensor:
        x = x_btc.transpose(1, 2).contiguous()      # [B, in_dim, T]
        x = self.stem(x)                            # [B, mid_dim, T]
        x = self.pre_ln(x.transpose(1, 2))           # [B, T, mid_dim]
        x = x.transpose(1, 2).contiguous()           # [B, mid_dim, T] -- MirrorDecoder wants channels-first
        return self.body(x).squeeze(1)               # [B, out_dim]


def load_native_proj_recon(ckpt_path: str, device: str = "cpu", arch: str = "conv1d"):
    """
    Load the backbone + proj_recon_decoder straight out of a native fairseq
    hydra_train checkpoint (a Step 2-style run). The backbone (including
    whatever post_extract_proj shape the checkpoint actually used — linear or
    mlp_gelu — is auto-detected by CheckpointLoader's own key inspection, so
    this reuses it unchanged rather than re-deriving the projection shape here.
    The decoder head is a _ProjMirrorHead (see above) fed from post_extract_proj's
    output — [B, T, 768], same shape as the transformer's own output.

    Returns (backbone, head, meta) — same shape as load_tr_recon.
    """
    from ..checkpoint_loader import CheckpointLoader

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not ("cfg" in raw and "model" in raw):
        raise ValueError(f"Not a native fairseq checkpoint: {ckpt_path}")
    sd = raw["model"]
    if not any(k.startswith("proj_recon_decoder.") for k in sd):
        raise ValueError(f"No proj_recon_decoder found in checkpoint: {ckpt_path}")

    print(f"[SignalRecon] native proj-recon backbone: {ckpt_path}")
    backbone = CheckpointLoader.from_file(ckpt_path, arch=arch)
    backbone.eval().to(device)

    dec_state = {k[len("proj_recon_decoder."):]: v for k, v in sd.items()
                 if k.startswith("proj_recon_decoder.")}
    embed_dim = backbone.config.hidden_size if hasattr(backbone, "config") else 768
    head = _ProjMirrorHead(in_dim=embed_dim, mid_dim=512, out_dim=245)
    r = head.load_state_dict(dec_state, strict=False)
    if r.missing_keys or r.unexpected_keys:
        raise RuntimeError(
            f"native proj-recon load error: missing={r.missing_keys} unexpected={r.unexpected_keys}"
        )
    head.eval().to(device)

    cfg = raw.get("cfg", {})
    model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else getattr(cfg, "model", cfg)
    lambda_recon_proj = (model_cfg.get("lambda_recon_proj") if isinstance(model_cfg, dict)
                          else getattr(model_cfg, "lambda_recon_proj", None))
    task_cfg = cfg.get("task", cfg) if isinstance(cfg, dict) else getattr(cfg, "task", cfg)
    normalize = (task_cfg.get("normalize") if isinstance(task_cfg, dict)
                 else getattr(task_cfg, "normalize", True))
    meta = {"lambda_recon_proj": lambda_recon_proj, "normalize": bool(normalize)}
    print(f"[SignalRecon] native proj-recon head loaded: {os.path.basename(ckpt_path)}  meta={meta}")
    return backbone, head, meta


def is_native_trans_recon_ckpt(ckpt_path: str) -> bool:
    """
    True if ckpt_path is a plain fairseq hydra_train checkpoint (keys: cfg/model)
    with a trans_recon_decoder attached (a Step 3-style transformer-reconstruction
    run) — the decoder lives inline as `trans_recon_decoder.*` keys alongside the
    full backbone, as opposed to load_tr_recon's separate transformer_mirror
    standalone-save format.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return (
        "cfg" in sd and "model" in sd
        and any(k.startswith("trans_recon_decoder.") for k in sd["model"])
    )


def load_native_trans_recon(ckpt_path: str, device: str = "cpu", arch: str = "conv1d"):
    """
    Load the backbone + trans_recon_decoder straight out of a native fairseq
    hydra_train checkpoint (a Step 3-style run). Architecturally identical to
    load_native_proj_recon's _ProjMirrorHead (both trans_recon_decoder and
    proj_recon_decoder are instantiated as the same _TransMirrorWrap class in
    data2vec_audio.py — stem Conv1d -> pre_ln -> MirrorReconDecoder body), just
    fed from a different point in the forward pass: the transformer encoder's
    own output, not post_extract_proj's pre-transformer output.

    Returns (backbone, head, meta) — same shape as load_tr_recon.
    """
    from ..checkpoint_loader import CheckpointLoader

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not ("cfg" in raw and "model" in raw):
        raise ValueError(f"Not a native fairseq checkpoint: {ckpt_path}")
    sd = raw["model"]
    if not any(k.startswith("trans_recon_decoder.") for k in sd):
        raise ValueError(f"No trans_recon_decoder found in checkpoint: {ckpt_path}")

    print(f"[SignalRecon] native trans-recon backbone: {ckpt_path}")
    backbone = CheckpointLoader.from_file(ckpt_path, arch=arch)
    backbone.eval().to(device)

    dec_state = {k[len("trans_recon_decoder."):]: v for k, v in sd.items()
                 if k.startswith("trans_recon_decoder.")}
    embed_dim = backbone.config.hidden_size if hasattr(backbone, "config") else 768
    head = _ProjMirrorHead(in_dim=embed_dim, mid_dim=512, out_dim=245)
    r = head.load_state_dict(dec_state, strict=False)
    if r.missing_keys or r.unexpected_keys:
        raise RuntimeError(
            f"native trans-recon load error: missing={r.missing_keys} unexpected={r.unexpected_keys}"
        )
    head.eval().to(device)

    cfg = raw.get("cfg", {})
    model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else getattr(cfg, "model", cfg)
    lambda_recon_trans = (model_cfg.get("lambda_recon_trans") if isinstance(model_cfg, dict)
                           else getattr(model_cfg, "lambda_recon_trans", None))
    task_cfg = cfg.get("task", cfg) if isinstance(cfg, dict) else getattr(cfg, "task", cfg)
    normalize = (task_cfg.get("normalize") if isinstance(task_cfg, dict)
                 else getattr(task_cfg, "normalize", True))
    meta = {"lambda_recon_trans": lambda_recon_trans, "normalize": bool(normalize)}
    print(f"[SignalRecon] native trans-recon head loaded: {os.path.basename(ckpt_path)}  meta={meta}")
    return backbone, head, meta


def load_tr_recon(ckpt_path: str, device: str = "cpu", arch: str = "conv1d",
                  backbone_ckpt: Optional[str] = None):
    """
    Load a transformer-recon checkpoint (keys: transformer_mirror / backbone_ckpt).
    The backbone comes from the fairseq checkpoint at `backbone_ckpt` (override with arg),
    loaded into the HF Data2VecAudioModel via CheckpointLoader (fairseq-free).
    Returns (backbone_model, mirror_head, meta).
    """
    from ..checkpoint_loader import CheckpointLoader

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "transformer_mirror" not in sd:
        raise ValueError(f"Not a TR-recon checkpoint (missing 'transformer_mirror'): {ckpt_path}")

    backbone_path = backbone_ckpt or sd.get("backbone_ckpt")
    if not backbone_path or not os.path.isfile(backbone_path):
        raise FileNotFoundError(f"TR-recon backbone not found: {backbone_path}")

    print(f"[SignalRecon] TR-recon backbone: {backbone_path}")
    backbone = CheckpointLoader.from_file(backbone_path, arch=arch)
    backbone.eval().to(device)

    mirror_sd = sd["transformer_mirror"]
    use_ln = any(k.startswith("pre_decoder_ln.") for k in mirror_sd)
    head = TransformerMirrorDecoder(
        encoder_embed_dim=int(sd.get("encoder_embed_dim", 768)),
        use_pre_decoder_ln=use_ln,
    )
    r = head.load_state_dict(mirror_sd, strict=False)
    if r.missing_keys or r.unexpected_keys:
        raise RuntimeError(
            f"TR-recon head load error: missing={r.missing_keys} unexpected={r.unexpected_keys}"
        )
    head.eval().to(device)

    meta = {k: sd.get(k) for k in ("n_samples", "steps", "lr", "warmup", "tag",
                                   "recon_path", "normalize")}
    meta["backbone_ckpt"] = backbone_path
    print(f"[SignalRecon] TR-recon head loaded: {os.path.basename(ckpt_path)}  "
          f"pre_decoder_ln={use_ln}  meta={meta}")
    return backbone, head, meta


def load_3ae_recon(ckpt_path: str, device: str = "cpu"):
    """
    Load a 3AE checkpoint (June-2026 train_reconstruction format, format>=5):
    one file holding the full backbone plus up to three reconstruction heads.

      data2vec_audio      fairseq-format backbone weights (remapped into HF model)
      fe_mirror           MirrorDecoder — decodes post-LN FE output      [B, 47, 512]
      proj_mirror         TransformerMirrorDecoder — decodes post_extract_proj [B, 47, 768]
      transformer_mirror  TransformerMirrorDecoder — decodes encoder output    [B, 47, 768]
      normalize           per-sample F.layer_norm flag used in training

    Returns (backbone, heads_dict, meta) — heads_dict maps 'fe'/'proj'/'tr' to modules
    (missing heads are absent from the dict).
    """
    from ..checkpoint_loader import load_3ae_backbone

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "data2vec_audio" not in sd:
        raise ValueError(f"Not a 3AE checkpoint (missing 'data2vec_audio'): {ckpt_path}")

    backbone = load_3ae_backbone(sd)
    backbone.eval().to(device)

    embed_dim = int(sd.get("encoder_embed_dim", 768))
    heads = {}
    if isinstance(sd.get("fe_mirror"), dict):
        head = MirrorDecoder()
        head.load_state_dict(sd["fe_mirror"], strict=True)
        heads["fe"] = head.eval().to(device)
    for key, name in (("proj_mirror", "proj"), ("transformer_mirror", "tr")):
        state = sd.get(key)
        if isinstance(state, dict):
            use_ln = any(k.startswith("pre_decoder_ln.") for k in state)
            head = TransformerMirrorDecoder(encoder_embed_dim=embed_dim, use_pre_decoder_ln=use_ln)
            head.load_state_dict(state, strict=True)
            heads[name] = head.eval().to(device)

    meta = {k: sd.get(k) for k in ("tag", "n_samples", "steps", "step_saved", "completed",
                                   "lr", "warmup", "normalize", "recon_path",
                                   "lambda_recon_fe", "lambda_recon_proj", "lambda_recon_trans")}
    print(f"[SignalRecon] 3AE loaded: {os.path.basename(ckpt_path)}  "
          f"heads={sorted(heads)}  normalize={meta['normalize']}  tag={meta.get('tag')}")
    return backbone, heads, meta


@torch.no_grad()
def _3ae_reconstruct_all(backbone, heads: dict, source: torch.Tensor) -> dict:
    """
    Single backbone forward, all three reconstruction pathways.
    Mirrors the training hooks: fe head reads feature_projection.layer_norm output,
    proj head reads feature_projection.projection output, tr head reads the encoder output.
    Returns {'fe'/'proj'/'tr': [B, 245]} for the heads present.
    """
    captured = {}
    handles = []
    if "fe" in heads:
        handles.append(backbone.feature_projection.layer_norm.register_forward_hook(
            lambda _m, _i, out: captured.__setitem__("fe_seq", out)))
    if "proj" in heads:
        handles.append(backbone.feature_projection.projection.register_forward_hook(
            lambda _m, _i, out: captured.__setitem__("proj_seq", out)))
    try:
        enc = backbone(input_values=source).last_hidden_state    # [B, T, 768]
    finally:
        for h in handles:
            h.remove()

    preds = {}
    if "fe" in heads:
        fe_seq = captured["fe_seq"]                              # [B, T, 512]
        preds["fe"] = heads["fe"](fe_seq.transpose(1, 2).contiguous()).squeeze(1)
    if "proj" in heads:
        preds["proj"] = heads["proj"](captured["proj_seq"])      # [B, 245]
    if "tr" in heads:
        preds["tr"] = heads["tr"](enc)
    return preds


@torch.no_grad()
def _fe_reconstruct(fe, ln, decoder, source: torch.Tensor) -> torch.Tensor:
    """input [B, 245] → FE → LN → MirrorDecoder → [B, 245]."""
    fe_out = fe(source)                      # [B, 512, 47]
    fe_ln = ln(fe_out.transpose(1, 2)).transpose(1, 2)
    pred = decoder(fe_ln)                    # [B, 1, 245]
    return pred.squeeze(1)


@torch.no_grad()
def _tr_reconstruct(backbone, head, source: torch.Tensor) -> torch.Tensor:
    """input [B, 245] → full backbone → TransformerMirrorDecoder → [B, 245]."""
    enc = backbone(input_values=source).last_hidden_state   # [B, T, 768]
    return head(enc)                                          # [B, 245]


@torch.no_grad()
def _proj_reconstruct(backbone, head, source: torch.Tensor) -> torch.Tensor:
    """
    input [B, 245] → FE → LN → proj → TransformerMirrorDecoder → [B, 245].
    Captures post_extract_proj's output via a forward hook on a full backbone
    pass (same pattern as _3ae_reconstruct_all's 'proj' pathway) rather than
    re-implementing the FE→LN→proj stack by hand.
    """
    captured = {}
    h = backbone.feature_projection.projection.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("proj_seq", out))
    try:
        backbone(input_values=source)
    finally:
        h.remove()
    return head(captured["proj_seq"])   # [B, 245]


# ── Dataset-level metrics (L1 per-component, L2 per-spectrum, L3 group summary) ─
#
# The sample-level view (a handful of overlay traces plus a mean MSE) cannot tell a
# working reconstruction from a degenerate one: a decoder that emits each sample's own
# mean value scores a respectable MSE while carrying no information at all. The metrics
# below are chosen so that failure mode, and the ones adjacent to it, are visible.
#
# All of them are computed against the SAME target the head was trained on — i.e. after
# the per-sample F.layer_norm when the checkpoint recorded `normalize` — so R² divides by
# the variance of that target, not of the raw wav.

# Head key → full legend text. Plots must use these strings, never the bare keys:
# a reader outside the project cannot be expected to know what 'proj' taps.
PATHWAY_LEGEND = {
    "fe":   "FE decoder — from the post-LayerNorm conv feature-extractor output "
            "[47×512], MirrorDecoder (~3.7M params)",
    "proj": "Projection decoder — from post_extract_proj, before the transformer "
            "[47×768], TransformerMirrorDecoder (~4.1M params)",
    "tr":   "Transformer decoder — from the transformer encoder output "
            "[47×768], TransformerMirrorDecoder (~4.1M params)",
}
PATHWAY_SHORT = {"fe": "FE decoder", "proj": "Projection decoder", "tr": "Transformer decoder"}

# Cross-head comparisons are confounded and every figure that makes one must say so.
CROSS_HEAD_CAVEAT = (
    "Cross-head comparison is confounded (TASKS.md T7): the FE decoder is a different "
    "architecture on a narrower input (47×512, MirrorDecoder) than the projection and "
    "transformer decoders (47×768, TransformerMirrorDecoder), so a gap between them mixes "
    "encoder information content with decoder capacity. Comparisons of one head across "
    "datasets are clean; comparisons between heads are not."
)


def _per_sample_metrics(target: np.ndarray, pred: np.ndarray, mse: np.ndarray) -> dict:
    """
    Per-sample skill metrics beyond MSE/MAE. `mse` is passed in rather than recomputed
    so the returned values stay consistent with the existing float32 F.mse_loss numbers.

    Returns numpy arrays, all [N]:
      r2         1 - MSE / var(target) — skill against predicting the sample's own mean.
                 0 means "no better than a flat line at the mean"; <0 means worse.
                 Landing at exactly 0 is a real outcome, not a bug - see _R2_BEAT_TOL.
      pearson    correlation of target and prediction — shape agreement, scale-invariant.
      amp_ratio  std(pred) / std(target) — <1 is dynamic-range collapse, the classic
                 autoencoder failure where output regresses toward the mean.
      peak_err   signed error at the target's argmax — fidelity at the spectral line.
    NaN where the quantity is undefined (constant target or constant prediction; several
    multi_channel components are flat or monotone).
    """
    t = np.asarray(target, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)

    var = t.var(axis=1)
    ok_var = var > 1e-12
    r2 = np.full(len(t), np.nan)
    r2[ok_var] = 1.0 - mse[ok_var] / var[ok_var]

    t_std, p_std = t.std(axis=1), p.std(axis=1)
    amp_ratio = np.where(ok_var, p_std / np.where(ok_var, t_std, 1.0), np.nan)

    tc = t - t.mean(axis=1, keepdims=True)
    pc = p - p.mean(axis=1, keepdims=True)
    denom = np.sqrt((tc ** 2).sum(axis=1) * (pc ** 2).sum(axis=1))
    pearson = np.where(denom > 1e-12, (tc * pc).sum(axis=1) / np.where(denom > 0, denom, 1.0), np.nan)

    peak_at = t.argmax(axis=1)
    at = np.arange(len(t))
    peak_err = p[at, peak_at] - t[at, peak_at]

    return {"r2": r2, "pearson": pearson, "amp_ratio": amp_ratio, "peak_err": peak_err}


# "Beating the baseline" needs a tolerance, not a bare `> 0`. A model that reproduces
# the baseline exactly (predicting each sample's mean) lands at R2 = 0 up to float32
# rounding, which a bare `> 0` would score as beating it on about half the samples.
_R2_BEAT_TOL = 1e-6


def _summary_table(rdf: pd.DataFrame, pathways: list) -> pd.DataFrame:
    """
    L3: one row per head, the headline numbers for a dataset.

    Median leads, mean follows. The mean is outlier-driven on the multi-component sets —
    the recorded T6 round has `samples` fe_mse mean 1.96 against a median of 0.36 — so a
    table that shows only the mean misrepresents the typical sample by a factor of five.
    """
    rows = []
    for k in pathways:
        mse, mae = rdf[f"{k}_mse"].to_numpy(), rdf[f"{k}_mae"].to_numpy()
        r2 = rdf[f"{k}_r2"].to_numpy()
        finite_r2 = r2[np.isfinite(r2)]
        rows.append({
            "head":              k,
            "head_label":        PATHWAY_SHORT.get(k, k),
            "n":                 len(rdf),
            "mse_median":        float(np.median(mse)),
            "mse_mean":          float(mse.mean()),
            "mse_p90":           float(np.percentile(mse, 90)),
            "mse_p99":           float(np.percentile(mse, 99)),
            "mae_median":        float(np.median(mae)),
            "mae_mean":          float(mae.mean()),
            "r2_median":         float(np.median(finite_r2)) if finite_r2.size else np.nan,
            "r2_mean":           float(finite_r2.mean()) if finite_r2.size else np.nan,
            "frac_r2_positive":  (float((finite_r2 > _R2_BEAT_TOL).mean())
                                  if finite_r2.size else np.nan),
            "pearson_median":    float(rdf[f"{k}_pearson"].median()),
            "amp_ratio_median":  float(rdf[f"{k}_amp_ratio"].median()),
            "peak_err_median":   float(rdf[f"{k}_peak_err"].median()),
        })
    return pd.DataFrame(rows)


def _spectrum_table(rdf: pd.DataFrame, pathways: list) -> pd.DataFrame:
    """
    L2, multi-component only: collapse each spectrum's components into one row.

    A multi-component sample is the set of files sharing (dataset_id, spec). Per-file
    error cannot distinguish "this whole spectrum reconstructs badly" from "one weak
    component in an otherwise fine spectrum" — mse_max and mse_spread separate them.

    Note the aggregation covers only the components that were actually drawn into this
    eval subset (n_comps_in_split), not all n_comps the dataset provides.
    """
    if "spec" not in rdf.columns or rdf["spec"].isna().all():
        return pd.DataFrame()
    agg = {}
    for k in pathways:
        agg[f"{k}_mse_mean"] = (f"{k}_mse", "mean")
        agg[f"{k}_mse_max"] = (f"{k}_mse", "max")
        agg[f"{k}_mse_min"] = (f"{k}_mse", "min")
    agg["n_comps_present"] = ("comp", "size")
    agg["n_comps"] = ("n_comps", "first")
    out = rdf.groupby(["dataset_id", "spec"], dropna=True).agg(**agg).reset_index()
    for k in pathways:
        out[f"{k}_mse_spread"] = out[f"{k}_mse_max"] - out[f"{k}_mse_min"]
    # Which component was the worst, for the head with the deepest tap available.
    ref = "tr" if "tr" in pathways else pathways[0]
    worst = (rdf.loc[rdf.groupby(["dataset_id", "spec"])[f"{ref}_mse"].idxmax(),
                     ["dataset_id", "spec", "comp"]]
                .rename(columns={"comp": "worst_comp"}))
    return out.merge(worst, on=["dataset_id", "spec"], how="left")


def _stratified_table(rdf: pd.DataFrame, pathways: list, axes: list) -> pd.DataFrame:
    """
    Median MSE (with quartiles and n) per bin of each stratification axis.

    Continuous axes are cut into quintiles; `comp` and `n_comps` are used as-is because
    they are categorical — component index is effectively an amplitude/shape class label
    (within one labeled_data spectrum, per-component std spans 0.027 to 0.33).
    Axes that cannot be binned (constant within this dataset) are skipped, and the caller
    is expected to say so on the figure rather than show an empty panel.
    """
    from ..signal_features import quantile_bins

    frames = []
    for axis in axes:
        if axis not in rdf.columns or rdf[axis].isna().all():
            continue
        values = rdf[axis].to_numpy(dtype=np.float64)
        uniq = np.unique(values[np.isfinite(values)])
        # Categorical only when the axis genuinely has few levels. peak_count is nominally
        # discrete but routinely takes 50+ distinct values on real spectra, which would
        # produce an unreadable 50-tick panel; quantile-bin those instead.
        categorical = (axis in ("comp", "n_comps", "n_comps_in_split", "peak_count")
                       and len(uniq) <= 12)
        if categorical:
            bins = values.copy()
            if len(uniq) < 2:
                continue
            labels = {float(u): f"{int(u)}" for u in uniq}
        else:
            idx, edges = quantile_bins(values, n_bins=5)
            if not edges:
                continue
            bins = np.where(idx >= 0, idx, np.nan).astype(float)
            labels = {float(i): edges[i] for i in range(len(edges))}

        for b in sorted(labels):
            sel = bins == b
            if sel.sum() < 3:                 # too few to quote a median
                continue
            row = {"axis": axis, "bin": b, "bin_label": labels[b], "n": int(sel.sum())}
            for k in pathways:
                v = rdf.loc[sel, f"{k}_mse"].to_numpy()
                row[f"{k}_mse_median"] = float(np.median(v))
                row[f"{k}_mse_q25"] = float(np.percentile(v, 25))
                row[f"{k}_mse_q75"] = float(np.percentile(v, 75))
            frames.append(row)
    return pd.DataFrame(frames)


def _profiles(target: np.ndarray, preds: dict) -> dict:
    """
    Position- and frequency-resolved error, averaged over the dataset.

    per_position_*   [L]   where along the 245 bins the error sits. Exposes conv-padding
                           edge artifacts and any period-5 structure from the final
                           (512, 5, 5) FE conv stage that maps 245 bins down to 47.
    mean_fft_*       [L//2+1] mean |rFFT| magnitude of target vs reconstruction. A decoder
                           that reproduces only the smooth envelope and loses narrow
                           spectral lines shows a magnitude deficit at high frequency
                           while its MSE still looks acceptable.
    """
    t = np.asarray(target, dtype=np.float64)
    prof = {
        "per_position_target_std": t.std(axis=0),
        "per_position_target_mean": t.mean(axis=0),
        "mean_fft_target": np.abs(np.fft.rfft(t, axis=1)).mean(axis=0),
        "freq_bins": np.fft.rfftfreq(t.shape[1], d=1.0),
    }
    for k, p in preds.items():
        p = np.asarray(p, dtype=np.float64)
        prof[f"per_position_abs_err_{k}"] = np.abs(p - t).mean(axis=0)
        prof[f"per_position_signed_err_{k}"] = (p - t).mean(axis=0)
        prof[f"mean_fft_pred_{k}"] = np.abs(np.fft.rfft(p, axis=1)).mean(axis=0)
    return prof


def run(
    df: pd.DataFrame,
    fe_ckpt: Optional[str] = None,
    tr_ckpt: Optional[str] = None,
    proj_ckpt: Optional[str] = None,
    recon_ckpt: Optional[str] = None,
    device: str = "cpu",
    arch: str = "conv1d",
    normalize: Optional[bool] = None,
    max_samples: int = 200,
    n_examples: int = 6,
    batch_size: int = 32,
    seed: int = 42,
    sample_meta: Optional[pd.DataFrame] = None,
    dataset_alias: str = "",
    dataset_subset: str = "",
) -> dict:
    """
    Run true signal reconstruction on df ('data' column of raw signals).

    Checkpoint options (either style):
      recon_ckpt  single 3AE checkpoint (data2vec_audio + fe_mirror/proj_mirror/
                  transformer_mirror) — runs ALL pathways it contains: FE, projection,
                  transformer.
      fe_ckpt     standalone FE autoencoder ckpt (encoder/layer_norm/decoder keys),
                  or a native fairseq checkpoint carrying an fe_recon_decoder.
      tr_ckpt     transformer-recon ckpt (transformer_mirror/backbone_ckpt keys)
      proj_ckpt   native fairseq checkpoint carrying a proj_recon_decoder (e.g. a
                  Step 2-style projection-reconstruction run)

    normalize: None → use the flag recorded in the checkpoint(s); True/False overrides.

    sample_meta: optional per-file component metadata from
      `data_loader.parse_component_metadata` (filename / dataset_id / comp / spec /
      n_comps). Its presence is what marks this dataset as multi-component and unlocks
      the per-spectrum (L2) table and the component-index stratifiers.
    dataset_alias / dataset_subset: labels only, carried into the figures so a reader can
      see which data a panel describes (e.g. 'in_dist' / 'single_channel_10k').

    Returns dict with:
      results_df    per-component (L1) metrics, one row per sample → recon_df.csv
      summary_df    per-head (L3) headline table, median-first
      spectrum_df   per-spectrum (L2) table — multi-component datasets only
      strat_df      median MSE per bin of each stratification axis
      profiles      per-position and per-frequency error arrays
      panel         the 6-example overlay data (unchanged)
      recon_{k}_mse_mean / _median, recon_{k}_mae_mean / _median  (unchanged scalars)
    """
    if not fe_ckpt and not tr_ckpt and not proj_ckpt and not recon_ckpt:
        return {"skipped": True,
                "error": "signal_reconstruction needs recon_ckpt (3AE) or fe_ckpt/tr_ckpt/proj_ckpt"}

    # ── Load models ───────────────────────────────────────────────────────────
    backbone_3ae = heads_3ae = meta_3ae = None
    fe_parts = tr_parts = proj_parts = None
    if recon_ckpt:
        backbone_3ae, heads_3ae, meta_3ae = load_3ae_recon(recon_ckpt, device=device)
    else:
        if fe_ckpt and is_native_fe_recon_ckpt(fe_ckpt):
            fe_parts = load_native_fe_recon(fe_ckpt, device=device)
        elif fe_ckpt:
            fe_parts = load_fe_recon(fe_ckpt, device=device)
        if tr_ckpt and is_native_trans_recon_ckpt(tr_ckpt):
            tr_parts = load_native_trans_recon(tr_ckpt, device=device, arch=arch)
        elif tr_ckpt:
            tr_parts = load_tr_recon(tr_ckpt, device=device, arch=arch)
        proj_parts = load_native_proj_recon(proj_ckpt, device=device, arch=arch) if proj_ckpt else None

    if normalize is None:
        flags = []
        if meta_3ae is not None:
            flags.append(bool(meta_3ae.get("normalize") or False))
        if fe_parts:
            flags.append(bool(fe_parts[3].get("normalize") or False))
        if tr_parts:
            flags.append(bool(tr_parts[2].get("normalize") or False))
        if proj_parts:
            flags.append(bool(proj_parts[2].get("normalize") or False))
        normalize = any(flags)
    print(f"[SignalRecon] normalize={normalize} (per-sample F.layer_norm)")

    # ── Data ──────────────────────────────────────────────────────────────────
    data = np.stack(df["data"].apply(np.array).values).astype(np.float32)
    fnames = (df["filename"].tolist() if "filename" in df.columns
              else [str(i) for i in range(len(df))])
    if len(data) > max_samples:
        idx = np.random.default_rng(seed).choice(len(data), max_samples, replace=False)
        idx.sort()
        data = data[idx]
        fnames = [fnames[i] for i in idx]

    target = torch.from_numpy(data)
    if normalize:
        target = F.layer_norm(target, target.shape[-1:])
    L = target.shape[1]
    target_np = target.numpy()

    # ── Reconstruct: preds keyed by pathway ('fe' / 'proj' / 'tr') ────────────
    pathway_names = (sorted(heads_3ae) if heads_3ae is not None
                     else [n for n, p in (("fe", fe_parts), ("tr", tr_parts), ("proj", proj_parts)) if p])
    preds = {k: torch.zeros_like(target) for k in pathway_names}
    for i in range(0, len(target), batch_size):
        batch = target[i : i + batch_size].to(device)
        if heads_3ae is not None:
            batch_preds = _3ae_reconstruct_all(backbone_3ae, heads_3ae, batch)
            for k, v in batch_preds.items():
                preds[k][i : i + batch_size] = v[..., :L].cpu()
        else:
            if fe_parts:
                fe, ln, decoder, _ = fe_parts
                preds["fe"][i : i + batch_size] = _fe_reconstruct(fe, ln, decoder, batch).cpu()
            if tr_parts:
                backbone, head, _ = tr_parts
                preds["tr"][i : i + batch_size] = _tr_reconstruct(backbone, head, batch)[..., :L].cpu()
            if proj_parts:
                backbone, head, _ = proj_parts
                preds["proj"][i : i + batch_size] = _proj_reconstruct(backbone, head, batch)[..., :L].cpu()

    # ── Metrics + results ─────────────────────────────────────────────────────
    rows = {"index": np.arange(len(target)), "filename": fnames}
    out = {
        "skipped": False,
        "normalize": normalize,
        "recon_ckpt": recon_ckpt,
        "fe_ckpt": fe_ckpt,
        "tr_ckpt": tr_ckpt,
        "proj_ckpt": proj_ckpt,
        "n_samples": len(target),
        "pathways": pathway_names,
    }
    if meta_3ae is not None:
        out["meta"] = meta_3ae
    if fe_parts:
        out["fe_meta"] = fe_parts[3]
    if tr_parts:
        out["tr_meta"] = tr_parts[2]
    if proj_parts:
        out["proj_meta"] = proj_parts[2]

    for k in pathway_names:
        mse = F.mse_loss(preds[k], target, reduction="none").mean(dim=1).numpy()
        mae = F.l1_loss(preds[k], target, reduction="none").mean(dim=1).numpy()
        rows[f"{k}_mse"] = mse
        rows[f"{k}_mae"] = mae
        out[f"recon_{k}_mse_mean"] = float(mse.mean())
        out[f"recon_{k}_mse_median"] = float(np.median(mse))
        # MAE alongside MSE: `data2vec_audio.py`'s recon_loss_type defaults to L1, not
        # L2/MSE, so a pathway trained under that default can show a large MSE (driven
        # by a few outlier errors L1 doesn't penalize as harshly) while still being a
        # genuinely improving, well-behaved fit by the metric it was actually trained
        # on. Report both rather than letting MSE alone look falsely catastrophic.
        out[f"recon_{k}_mae_mean"] = float(mae.mean())
        out[f"recon_{k}_mae_median"] = float(np.median(mae))

        # Skill metrics beyond error magnitude — see _per_sample_metrics.
        for name, values in _per_sample_metrics(
                target_np, preds[k].numpy(), mse).items():
            rows[f"{k}_{name}"] = values

    rdf = pd.DataFrame(rows)

    # ── Sample descriptors + component metadata → stratification axes ──────────
    rdf = pd.concat([rdf, compute_signal_features(target_np)], axis=1)

    if sample_meta is not None and len(sample_meta):
        meta_cols = [c for c in sample_meta.columns if c != "filename"]
        rdf = rdf.merge(sample_meta[["filename"] + meta_cols], on="filename", how="left")
        matched = int(rdf["comp"].notna().sum())
        if matched == 0:
            print(f"[SignalRecon] WARNING: sample_meta matched 0 of {len(rdf)} filenames "
                  f"— falling back to single-component treatment")
        else:
            print(f"[SignalRecon] component metadata matched {matched}/{len(rdf)} samples")
    is_multi = "comp" in rdf.columns and rdf["comp"].notna().any()
    if "component_group" not in rdf.columns:
        rdf["component_group"] = "single"
    rdf["component_group"] = rdf["component_group"].fillna("single")

    out["results_df"] = rdf
    out["component_group"] = "multi" if is_multi else "single"
    out["dataset_alias"] = dataset_alias
    out["dataset_subset"] = dataset_subset

    # ── L3 summary, L2 per-spectrum, stratified medians, profiles ─────────────
    out["summary_df"] = _summary_table(rdf, pathway_names)

    strat_axes = list(STRATIFIER_ORDER)
    if is_multi:
        out["spectrum_df"] = _spectrum_table(rdf, pathway_names)
        strat_axes += ["comp", "n_comps"]
    out["strat_df"] = _stratified_table(rdf, pathway_names, strat_axes)

    out["profiles"] = _profiles(target_np, {k: preds[k].numpy() for k in pathway_names})

    # Full arrays for the hexbin / calibration figures. ~1 MB per pathway at n=1000;
    # nothing serializes the results dict, and the CSV pass only touches DataFrames.
    out["_arrays"] = {"target": target_np,
                      "preds": {k: preds[k].numpy() for k in pathway_names}}

    # Example panel for the overlay plot (deterministic pick)
    ex_idx = np.linspace(0, len(target) - 1, min(n_examples, len(target)), dtype=int)
    out["panel"] = {
        "indices": ex_idx.tolist(),
        "names": [fnames[i] for i in ex_idx],
        "target": target[ex_idx].numpy(),
        **{f"pred_{k}": preds[k][ex_idx].numpy() for k in pathway_names},
    }

    label = dataset_alias or dataset_subset or "dataset"
    msg = [f"[SignalRecon] {label} ({out['component_group']}-component) n={len(target)}"]
    for k in pathway_names:
        s = out["summary_df"].set_index("head").loc[k]
        msg.append(f"{k.upper()} MSE med={s['mse_median']:.4e} mean={s['mse_mean']:.4e} "
                   f"R2med={s['r2_median']:.3f} ampRatio={s['amp_ratio_median']:.3f}")
    print("  ".join(msg))
    return out
