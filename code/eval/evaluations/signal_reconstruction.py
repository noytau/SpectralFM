"""
Signal Reconstruction Evaluation (true reconstruction through the full pipeline)
---------------------------------------------------------------------------------
Ports the reconstruction flow from compare_fe_vs_trans_recon.py / train_reconstruction.py
(branch: reconstruction-loss-experiments) into the fairseq-free eval package.

Two reconstruction pathways, each loaded from its OWN checkpoint:

  FE recon         : input → FE → LayerNorm → MirrorDecoder → signal
                     checkpoint keys: encoder / layer_norm / decoder
                     (e.g. autoencoder_experiments/fe_signal_recon_*/ckpt_*.pt)

  Transformer recon: input → FE → LN → proj → Transformer → TransformerMirrorDecoder → signal
                     checkpoint keys: transformer_mirror / backbone_ckpt
                     The backbone (FE+LN+proj+transformer) is loaded from the fairseq
                     checkpoint referenced by `backbone_ckpt` (remapped into the HF
                     Data2VecAudioModel — no fairseq import needed).
                     (e.g. autoencoder_experiments/transformer_recon_*/ckpt_tr_*.pt)

Either or both pathways can run — pass fe_ckpt and/or tr_ckpt.

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


def run(
    df: pd.DataFrame,
    fe_ckpt: Optional[str] = None,
    tr_ckpt: Optional[str] = None,
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
      fe_ckpt     standalone FE autoencoder ckpt (encoder/layer_norm/decoder keys)
      tr_ckpt     transformer-recon ckpt (transformer_mirror/backbone_ckpt keys)

    normalize: None → use the flag recorded in the checkpoint(s); True/False overrides.

    Returns dict with per-sample results_df (full CSV export), mean MSEs per pathway
    (recon_fe_mse_mean / recon_proj_mse_mean / recon_tr_mse_mean), and an example
    panel (targets + reconstructions) for the overlay plot.
    """
    if not fe_ckpt and not tr_ckpt and not recon_ckpt:
        return {"skipped": True,
                "error": "signal_reconstruction needs recon_ckpt (3AE) or fe_ckpt/tr_ckpt"}

    # ── Load models ───────────────────────────────────────────────────────────
    backbone_3ae = heads_3ae = meta_3ae = None
    fe_parts = tr_parts = None
    if recon_ckpt:
        backbone_3ae, heads_3ae, meta_3ae = load_3ae_recon(recon_ckpt, device=device)
    else:
        fe_parts = load_fe_recon(fe_ckpt, device=device) if fe_ckpt else None
        tr_parts = load_tr_recon(tr_ckpt, device=device, arch=arch) if tr_ckpt else None

    if normalize is None:
        flags = []
        if meta_3ae is not None:
            flags.append(bool(meta_3ae.get("normalize") or False))
        if fe_parts:
            flags.append(bool(fe_parts[3].get("normalize") or False))
        if tr_parts:
            flags.append(bool(tr_parts[2].get("normalize") or False))
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
                     else [n for n, p in (("fe", fe_parts), ("tr", tr_parts)) if p])
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

    # ── Metrics + results ─────────────────────────────────────────────────────
    rows = {"index": np.arange(len(target)), "filename": fnames}
    out = {
        "skipped": False,
        "normalize": normalize,
        "recon_ckpt": recon_ckpt,
        "fe_ckpt": fe_ckpt,
        "tr_ckpt": tr_ckpt,
        "n_samples": len(target),
        "pathways": pathway_names,
    }
    if meta_3ae is not None:
        out["meta"] = meta_3ae
    if fe_parts:
        out["fe_meta"] = fe_parts[3]
    if tr_parts:
        out["tr_meta"] = tr_parts[2]

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
