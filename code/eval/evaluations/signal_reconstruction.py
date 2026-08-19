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
                     checkpoint keys: transformer_mirror / backbone_ckpt
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


def load_native_proj_recon(ckpt_path: str, device: str = "cpu", arch: str = "conv1d"):
    """
    Load the backbone + proj_recon_decoder straight out of a native fairseq
    hydra_train checkpoint (a Step 2-style run). The backbone (including
    whatever post_extract_proj shape the checkpoint actually used — linear or
    mlp_gelu — is auto-detected by CheckpointLoader's own key inspection, so
    this reuses it unchanged rather than re-deriving the projection shape here.
    The decoder itself is architecturally identical to trans_recon_decoder's
    TransformerMirrorDecoder (post_extract_proj output is [B, T, 768], same
    shape as the transformer's own output — see data2vec_audio.py), just
    fed from a different point in the forward pass (pre-transformer).

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
    use_ln = any(k.startswith("pre_decoder_ln.") for k in dec_state)
    head = TransformerMirrorDecoder(encoder_embed_dim=embed_dim, use_pre_decoder_ln=use_ln)
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
    meta = {"lambda_recon_proj": lambda_recon_proj, "normalize": bool(normalize),
            "pre_decoder_ln": use_ln}
    print(f"[SignalRecon] native proj-recon head loaded: {os.path.basename(ckpt_path)}  meta={meta}")
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

    Returns dict with per-sample results_df (full CSV export), mean MSEs per pathway
    (recon_fe_mse_mean / recon_proj_mse_mean / recon_tr_mse_mean), and an example
    panel (targets + reconstructions) for the overlay plot.
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
        tr_parts = load_tr_recon(tr_ckpt, device=device, arch=arch) if tr_ckpt else None
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
        rows[f"{k}_mse"] = mse
        out[f"recon_{k}_mse_mean"] = float(mse.mean())
        out[f"recon_{k}_mse_median"] = float(np.median(mse))

    out["results_df"] = pd.DataFrame(rows)

    # Example panel for the overlay plot (deterministic pick)
    ex_idx = np.linspace(0, len(target) - 1, min(n_examples, len(target)), dtype=int)
    out["panel"] = {
        "indices": ex_idx.tolist(),
        "names": [fnames[i] for i in ex_idx],
        "target": target[ex_idx].numpy(),
        **{f"pred_{k}": preds[k][ex_idx].numpy() for k in pathway_names},
    }

    msg = [f"[SignalRecon] n={len(target)}"]
    for k in pathway_names:
        msg.append(f"{k.upper()} MSE mean={out[f'recon_{k}_mse_mean']:.4e}")
    print("  ".join(msg))
    return out
