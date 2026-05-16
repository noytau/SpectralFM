"""
Per-component init, freeze, audit, param-grouping, and logging for the
``train_reconstruction.py`` transformer-AE setup (FE + LayerNorm + post_extract_proj +
TransformerEncoder + TransformerMirrorDecoder head).

Three checkpoint flavors are supported as init sources (auto-detected by key prefixes):

  * ``fairseq_audio``  - SpectralFM ``Data2VecAudioModel`` ``.pt`` (e.g. Feb-25):
        ``feature_extractor.*``, ``layer_norm.*``, ``post_extract_proj.*``, ``encoder.*``
  * ``data2vec_multi`` - base_libri-style ``.pt``:
        conv FE in ``modality_encoders.AUDIO.local_encoder.*`` (**shape-incompatible** with
        SpectralFM's 5-layer stack -> FE merge will skip everything),
        post-FE LN at ``modality_encoders.AUDIO.project_features.1.*``,
        post-extract projection at ``modality_encoders.AUDIO.project_features.2.*``,
        ViT-style transformer blocks at ``blocks.X.*`` (split-QKV remapped to fairseq).
  * ``apr28_fe_recon`` - FE-AE ``ckpt_{tag}.pt`` from ``--recon_path fe``:
        ``encoder.*`` (= conv FE state dict, **fits**), ``layer_norm.*``, ``decoder.*``.

Modules ``mask_emb``, ``final_proj``, ``fe_recon_decoder``, ``trans_recon_decoder``,
and the EMA exist on ``Data2VecAudioModel`` but are **never touched** when called via
``features_only=True, mask=False`` (the ``train_reconstruction`` forward); we freeze
them to keep gradient flow clean and the trainable-param count honest.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


# ────────────────────────────────────────────────────────────────────────────
#  Checkpoint loading helpers
# ────────────────────────────────────────────────────────────────────────────

def _read_state_dict(ckpt_path: str) -> dict:
    """Return a flat ``{full_dotted_key: tensor}`` view of the checkpoint.

    Handles three on-disk layouts:
      * fairseq audio: ``state["model"]`` is the flat state dict (e.g. Feb-25).
      * data2vec_multi: top-level dict is already flat (e.g. base_libri).
      * apr28 FE-recon: top-level keys ``encoder``/``layer_norm``/``decoder`` hold
        **nested** sub-state-dicts → unpack so ``encoder`` becomes ``encoder.<sub>``.
    """
    if not ckpt_path or str(ckpt_path).lower() == "none":
        return {}
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and isinstance(raw.get("model"), dict):
        sd = dict(raw["model"])
    elif isinstance(raw, dict):
        sd = dict(raw)
    else:
        return {}
    sd.pop("_ema", None)

    # Unpack any top-level nested dicts into ``{top}.{sub}`` flat keys.
    flat: dict = {}
    for k, v in sd.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                if torch.is_tensor(sv):
                    flat[f"{k}.{sk}"] = sv
        elif torch.is_tensor(v):
            flat[k] = v
        # Drop non-tensor scalars (tag, losses, etc.).
    return flat


def _strip_prefix(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def detect_ckpt_flavor(sd: dict) -> str:
    """Best-effort flavor detection from top-level key prefixes."""
    if any(k.startswith("feature_extractor.") for k in sd):
        return "fairseq_audio"
    if any(k.startswith("modality_encoders.") or k.startswith("blocks.") for k in sd):
        return "data2vec_multi"
    if any(k.startswith("encoder.") for k in sd) and "decoder" in sd or all(
        k.split(".")[0] in {"encoder", "layer_norm", "decoder", "losses", "lrs", "tag", "lr",
                            "warmup", "n_samples", "steps", "batch_size", "grad_accum_steps",
                            "micro_batch_size", "effective_batch_size", "backbone_ckpt"}
        for k in sd
    ):
        return "apr28_fe_recon"
    return "unknown"


def _shape_safe_merge(target: nn.Module, src_sd: dict, *, label: str) -> Tuple[int, int, int]:
    """Load only tensors whose shape matches ``target.state_dict()``.

    Returns ``(loaded, skipped_shape, missing_in_target)``.
    """
    tgt = target.state_dict()
    loaded = {k: v for k, v in src_sd.items() if k in tgt and v.shape == tgt[k].shape}
    skipped_shape = sum(1 for k, v in src_sd.items() if k in tgt and v.shape != tgt[k].shape)
    missing = sum(1 for k in tgt if k not in src_sd)
    if loaded:
        target.load_state_dict(loaded, strict=False)
    print(
        f"    [{label}] loaded={len(loaded)}/{len(tgt)}  "
        f"shape-skip={skipped_shape}  src-extra={len(src_sd) - len(loaded) - skipped_shape}  "
        f"target-missing={missing}"
    )
    return len(loaded), skipped_shape, missing


# ── Conv FE ────────────────────────────────────────────────────────────────

def _fe_subdict(sd: dict) -> Tuple[dict, str]:
    """Pull a conv-FE sub-state-dict out of any supported flavor."""
    for prefix in ("feature_extractor.", "modality_encoders.AUDIO.local_encoder.", "encoder."):
        sub = _strip_prefix(sd, prefix)
        if sub:
            return sub, prefix
    return {}, ""


def load_fe_from_ckpt(model, ckpt_path: str) -> dict:
    sd = _read_state_dict(ckpt_path)
    sub, src = _fe_subdict(sd)
    if not sub:
        print(f"  [fe ] no FE keys found in {ckpt_path}")
        return {"source": ckpt_path, "src_prefix": "", "loaded": 0}
    print(f"  [fe ] source prefix='{src}' from {ckpt_path}")
    n, _, _ = _shape_safe_merge(model.feature_extractor, sub, label="feature_extractor")
    return {"source": ckpt_path, "src_prefix": src, "loaded": n}


# ── Post-FE LayerNorm (the 512-d LN) ───────────────────────────────────────

def _ln_subdict(sd: dict) -> Tuple[dict, str]:
    for prefix in ("layer_norm.", "modality_encoders.AUDIO.project_features.1."):
        sub = _strip_prefix(sd, prefix)
        if sub:
            return sub, prefix
    return {}, ""


def load_layer_norm_from_ckpt(model, ckpt_path: str) -> dict:
    sd = _read_state_dict(ckpt_path)
    sub, src = _ln_subdict(sd)
    if not sub:
        print(f"  [ln ] no LN keys found in {ckpt_path}")
        return {"source": ckpt_path, "src_prefix": "", "loaded": 0}
    print(f"  [ln ] source prefix='{src}' from {ckpt_path}")
    n, _, _ = _shape_safe_merge(model.layer_norm, sub, label="layer_norm")
    return {"source": ckpt_path, "src_prefix": src, "loaded": n}


# ── post_extract_proj (512 -> encoder_embed_dim) ───────────────────────────

def _proj_subdict(sd: dict) -> Tuple[dict, str]:
    for prefix in ("post_extract_proj.", "modality_encoders.AUDIO.project_features.2."):
        sub = _strip_prefix(sd, prefix)
        if sub:
            return sub, prefix
    return {}, ""


def load_post_extract_proj_from_ckpt(model, ckpt_path: str) -> dict:
    sd = _read_state_dict(ckpt_path)
    sub, src = _proj_subdict(sd)
    if not sub:
        print(f"  [pj ] no post_extract_proj keys found in {ckpt_path}")
        return {"source": ckpt_path, "src_prefix": "", "loaded": 0}
    print(f"  [pj ] source prefix='{src}' from {ckpt_path}")
    proj = model.post_extract_proj
    if isinstance(proj, nn.Linear):
        n, _, _ = _shape_safe_merge(proj, sub, label="post_extract_proj")
    else:
        n, _, _ = _shape_safe_merge(proj, sub, label="post_extract_proj")
    return {"source": ckpt_path, "src_prefix": src, "loaded": n}


# ── Transformer encoder (with ViT->fairseq remap for data2vec_multi) ───────

def _remap_vit_blocks_to_fairseq(blocks_sd: dict, embed_dim: int = 768) -> dict:
    """Map ``blocks.X.*`` (ViT/timm) -> ``layers.X.*`` (fairseq TransformerEncoder).

    QKV is stored fused in ViT; fairseq stores separate q/k/v projections.
    """
    out: dict = {}
    e = int(embed_dim)
    for k, v in blocks_sd.items():
        parts = k.split(".")
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        rest = ".".join(parts[1:])
        base = f"layers.{idx}"

        if rest == "attn.qkv.weight" and v.shape == (3 * e, e):
            q, k_, v_ = v[:e], v[e:2 * e], v[2 * e:]
            out[f"{base}.self_attn.q_proj.weight"] = q
            out[f"{base}.self_attn.k_proj.weight"] = k_
            out[f"{base}.self_attn.v_proj.weight"] = v_
        elif rest == "attn.qkv.bias" and v.shape == (3 * e,):
            q, k_, v_ = v[:e], v[e:2 * e], v[2 * e:]
            out[f"{base}.self_attn.q_proj.bias"] = q
            out[f"{base}.self_attn.k_proj.bias"] = k_
            out[f"{base}.self_attn.v_proj.bias"] = v_
        elif rest == "attn.proj.weight":
            out[f"{base}.self_attn.out_proj.weight"] = v
        elif rest == "attn.proj.bias":
            out[f"{base}.self_attn.out_proj.bias"] = v
        elif rest == "mlp.fc1.weight":
            out[f"{base}.fc1.weight"] = v
        elif rest == "mlp.fc1.bias":
            out[f"{base}.fc1.bias"] = v
        elif rest == "mlp.fc2.weight":
            out[f"{base}.fc2.weight"] = v
        elif rest == "mlp.fc2.bias":
            out[f"{base}.fc2.bias"] = v
        elif rest == "norm1.weight":
            out[f"{base}.self_attn_layer_norm.weight"] = v
        elif rest == "norm1.bias":
            out[f"{base}.self_attn_layer_norm.bias"] = v
        elif rest == "norm2.weight":
            out[f"{base}.final_layer_norm.weight"] = v
        elif rest == "norm2.bias":
            out[f"{base}.final_layer_norm.bias"] = v
    return out


def load_transformer_from_ckpt(model, ckpt_path: str) -> dict:
    sd = _read_state_dict(ckpt_path)
    enc_sub = _strip_prefix(sd, "encoder.")
    blocks_sub = _strip_prefix(sd, "blocks.")
    embed = int(getattr(model.cfg, "encoder_embed_dim", 768))
    n_layers_tgt = int(getattr(model.cfg, "encoder_layers", 12))

    if enc_sub:
        print(f"  [tr ] source prefix='encoder.' (fairseq layout) from {ckpt_path}")
        n, _, _ = _shape_safe_merge(model.encoder, enc_sub, label="encoder")
        return {"source": ckpt_path, "src_prefix": "encoder.", "loaded": n,
                "src_layers": _count_block_layers(enc_sub, "layers."), "tgt_layers": n_layers_tgt}

    if blocks_sub:
        print(f"  [tr ] source prefix='blocks.' (data2vec_multi/ViT) from {ckpt_path}")
        src_layers = _count_block_layers(blocks_sub, "")
        if src_layers < n_layers_tgt:
            print(f"    [warn] source has {src_layers} blocks but target encoder has "
                  f"{n_layers_tgt} layers; only first {src_layers} layers initialized.")
        remapped = _remap_vit_blocks_to_fairseq(blocks_sub, embed_dim=embed)
        n, _, _ = _shape_safe_merge(model.encoder, remapped, label="encoder (remap)")
        return {"source": ckpt_path, "src_prefix": "blocks.", "loaded": n,
                "src_layers": src_layers, "tgt_layers": n_layers_tgt}

    print(f"  [tr ] no transformer keys found in {ckpt_path}")
    return {"source": ckpt_path, "src_prefix": "", "loaded": 0,
            "src_layers": 0, "tgt_layers": n_layers_tgt}


def _count_block_layers(sd: dict, prefix: str) -> int:
    idxs = set()
    for k in sd:
        rest = k[len(prefix):] if prefix and k.startswith(prefix) else k
        parts = rest.split(".")
        if parts and parts[0].isdigit():
            idxs.add(int(parts[0]))
    return len(idxs)


# ── Head (TransformerMirrorDecoder) ────────────────────────────────────────

def load_head_from_ckpt(head, ckpt_path: str) -> dict:
    """Load a reconstruction head (``TransformerMirrorDecoder`` or ``MirrorDecoder``).

    Recognised source layouts (auto-detect by key prefix on the *flat* state dict
    produced by :func:`_read_state_dict`):

      * ``transformer_mirror.*``  → train_reconstruction.py transformer-AE checkpoint
      * ``fe_mirror.*``           → train_reconstruction.py 2-AE checkpoint (head_fe)
      * ``decoder.*``             → Apr-28 FE-AE checkpoint (encoder/layer_norm/decoder)
                                    or transformer-AE inner ``transformer_mirror.decoder.*``
    """
    sd_full = _read_state_dict(ckpt_path)
    if not sd_full:
        return {"source": ckpt_path, "src_prefix": "", "loaded": 0}

    # Candidate prefixes in priority order; check both the full head key set and
    # just the "decoder.*" sub-keys (for MirrorDecoder, which has no stem/LN).
    tgt_keys = set(head.state_dict().keys())
    has_stem = any(k.startswith("stem.") for k in tgt_keys)
    candidates = []
    if has_stem:
        # TransformerMirrorDecoder: prefer fully-prefixed head sources.
        candidates += [("transformer_mirror.", "transformer_mirror"),
                       ("fe_mirror.",          "fe_mirror")]
    else:
        # MirrorDecoder: keys live under "decoder.*" in saved heads,
        # or are bare "layers.*" in the FE-AE format.
        candidates += [("transformer_mirror.decoder.", "transformer_mirror.decoder"),
                       ("fe_mirror.decoder.",          "fe_mirror.decoder"),
                       ("decoder.",                    "decoder (FE-AE format)")]

    src: dict = {}
    label = ""
    for prefix, lab in candidates:
        sub = _strip_prefix(sd_full, prefix)
        if sub:
            src, label = sub, lab
            break
    if not src:
        print(f"  [hd ] no head keys found in {ckpt_path}")
        return {"source": ckpt_path, "src_prefix": "", "loaded": 0}
    print(f"  [hd ] source='{label}' from {ckpt_path}")
    n, _, _ = _shape_safe_merge(head, src, label="head")
    return {"source": ckpt_path, "src_prefix": label, "loaded": n}


# ────────────────────────────────────────────────────────────────────────────
#  Component view: stable names mapped to nn.Module(s)
# ────────────────────────────────────────────────────────────────────────────

def _component_modules(
    backbone, head_trans, head_fe: Optional[nn.Module] = None
) -> List[Tuple[str, nn.Module]]:
    """Stable (name, module) list for the transformer-AE setup.

    ``head_trans`` is the existing :class:`TransformerMirrorDecoder` (transformer-output
    head). ``head_fe`` is the optional second :class:`MirrorDecoder` that branches off
    post-LN FE features when ``lambda_recon_fe > 0`` (the 2-AE setup).
    """
    comps: List[Tuple[str, nn.Module]] = [
        ("fe",          backbone.feature_extractor),
        ("ln",          backbone.layer_norm),
        ("proj",        backbone.post_extract_proj),
        ("transformer", backbone.encoder),
        ("head_trans",  head_trans),
    ]
    if head_fe is not None:
        comps.append(("head_fe", head_fe))
    return comps


# ────────────────────────────────────────────────────────────────────────────
#  Freeze unused submodules + per-component freeze toggle
# ────────────────────────────────────────────────────────────────────────────

# Submodules that exist on Data2VecAudioModel but are NOT touched in our forward
# (features_only=True, mask=False). Freezing them keeps trainable counts accurate
# and stops them from being added to optimizer param groups.
_UNUSED_TOP_LEVEL = ("final_proj", "fe_recon_decoder", "trans_recon_decoder", "mask_emb")


def freeze_unused_d2v_submodules(backbone) -> Tuple[int, List[str]]:
    """Freeze submodules that are dead under ``features_only=True, mask=False``."""
    frozen = 0
    names: List[str] = []
    for n in _UNUSED_TOP_LEVEL:
        mod = getattr(backbone, n, None)
        if mod is None:
            continue
        if isinstance(mod, nn.Parameter):
            if mod.requires_grad:
                mod.requires_grad = False
                frozen += mod.numel()
                names.append(n)
        elif isinstance(mod, nn.Module):
            for p in mod.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    frozen += p.numel()
            if any(True for _ in mod.parameters()):
                names.append(n)
    if hasattr(backbone, "ema") and backbone.ema is not None:
        for p in backbone.ema.model.parameters() if hasattr(backbone.ema, "model") else []:
            if p.requires_grad:
                p.requires_grad = False
                frozen += p.numel()
        names.append("ema")
    return frozen, names


def apply_freezes(
    backbone, head_trans, head_fe=None, *, freeze: Dict[str, bool]
) -> Dict[str, int]:
    """Freeze components by name; returns per-component frozen-param counts.

    Accepts ``head`` (legacy) or ``head_trans``/``head_fe`` keys in ``freeze``.
    """
    out: Dict[str, int] = {}
    comps = dict(_component_modules(backbone, head_trans, head_fe))
    # Back-compat: map legacy "head" → "head_trans".
    freeze = dict(freeze)
    if "head" in freeze and "head_trans" not in freeze:
        freeze["head_trans"] = freeze.pop("head")
    for name, do_freeze in freeze.items():
        if not do_freeze:
            continue
        mod = comps.get(name)
        if mod is None:
            continue
        n = 0
        for p in mod.parameters():
            if p.requires_grad:
                p.requires_grad = False
                n += p.numel()
        out[name] = n
        if name == "fe":
            backbone.feature_grad_mult = 0.0
    return out


# ────────────────────────────────────────────────────────────────────────────
#  Per-component LR groups + cosine schedule per group
# ────────────────────────────────────────────────────────────────────────────

def build_param_groups(
    backbone, head_trans, head_fe=None, *, lr_base: float,
    lr_overrides: Dict[str, Optional[float]],
) -> List[dict]:
    """Build Adam param groups, one per component, skipping all-frozen modules.

    Each group has a ``base_lr`` attached for the cosine scheduler to read.
    Back-compat: ``lr_overrides["head"]`` (if present) is treated as
    ``lr_overrides["head_trans"]``.
    """
    lr_overrides = dict(lr_overrides)
    if "head" in lr_overrides and "head_trans" not in lr_overrides:
        lr_overrides["head_trans"] = lr_overrides.pop("head")
    groups: List[dict] = []
    for name, mod in _component_modules(backbone, head_trans, head_fe):
        params = [p for p in mod.parameters() if p.requires_grad]
        if not params:
            continue
        lr = lr_overrides.get(name)
        lr_g = float(lr) if lr is not None else float(lr_base)
        groups.append({"params": params, "lr": lr_g, "base_lr": lr_g, "name": name})
    return groups


def apply_cosine_to_groups(
    groups: List[dict], step: int, total_steps: int, warmup_steps: int
) -> Dict[str, float]:
    """Apply cosine(warmup) schedule **per group** using each group's ``base_lr``."""
    import math
    if step <= warmup_steps:
        factor = step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    out: Dict[str, float] = {}
    for g in groups:
        lr_now = float(g["base_lr"]) * factor
        g["lr"] = lr_now
        out[g["name"]] = lr_now
    return out


# ────────────────────────────────────────────────────────────────────────────
#  Audit: who's trainable, who's actually receiving gradients
# ────────────────────────────────────────────────────────────────────────────

def component_param_summary(backbone, head_trans, head_fe=None) -> List[dict]:
    """One row per component: total / trainable / frozen / pct_trainable."""
    rows: List[dict] = []
    for name, mod in _component_modules(backbone, head_trans, head_fe):
        total = sum(p.numel() for p in mod.parameters())
        train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        rows.append({
            "name": name,
            "class": type(mod).__name__,
            "total": total,
            "trainable": train,
            "frozen": total - train,
        })
    return rows


def print_component_summary(rows: List[dict], *, tag: str = "components") -> None:
    print(f"\n  ── {tag} ──")
    print(f"    {'name':<12} {'class':<28} {'total':>14} {'trainable':>14} {'frozen':>14}")
    for r in rows:
        print(f"    {r['name']:<12} {r['class']:<28} {r['total']:>14,} {r['trainable']:>14,} {r['frozen']:>14,}")
    tot = sum(r["total"] for r in rows)
    tr = sum(r["trainable"] for r in rows)
    print(f"    {'TOTAL':<12} {'-':<28} {tot:>14,} {tr:>14,} {tot - tr:>14,}")


def audit_gradient_flow(
    backbone, head_trans, source_template: torch.Tensor, *,
    micro_bs: int = 4, head_fe=None, lambda_trans: float = 1.0,
    lambda_fe: float = 0.0,
) -> dict:
    """Run a single forward+backward on a tiny batch and check which trainable
    parameters actually received gradients.

    Mirrors the production loss formula: ``λ_trans · recon_trans + λ_fe · recon_fe``
    with the FE branch taking post-LN features via a forward hook on
    ``backbone.layer_norm``.
    """
    import torch.nn.functional as F
    bb_was = backbone.training
    ht_was = head_trans.training
    hf_was = head_fe.training if head_fe is not None else None
    backbone.train(); head_trans.train()
    if head_fe is not None:
        head_fe.train()
    src = source_template[:micro_bs].clone().detach().to(next(head_trans.parameters()).device)
    target = src[:, :245].float()

    captured: Dict[str, torch.Tensor] = {}
    def _ln_hook(_mod, _inp, out):
        captured["fe_seq"] = out
    h = backbone.layer_norm.register_forward_hook(_ln_hook) if head_fe is not None else None
    try:
        out = backbone(src, padding_mask=None, mask=False, features_only=True)
        loss = torch.zeros((), device=src.device)
        if lambda_trans > 0:
            recon_t, _ = head_trans(out["x"])
            loss = loss + lambda_trans * F.mse_loss(recon_t.squeeze(1), target)
        if lambda_fe > 0 and head_fe is not None and "fe_seq" in captured:
            fe_seq = captured["fe_seq"]
            recon_f, _ = head_fe(fe_seq.transpose(1, 2))
            loss = loss + lambda_fe * F.mse_loss(recon_f.squeeze(1), target)
        if loss.requires_grad:
            loss.backward()
    finally:
        if h is not None:
            h.remove()

    no_grad_named: List[Tuple[str, int]] = []
    yes_grad_named: List[Tuple[str, int]] = []
    iterables = list(backbone.named_parameters()) + [
        (f"head_trans.{n}", p) for n, p in head_trans.named_parameters()
    ]
    if head_fe is not None:
        iterables += [(f"head_fe.{n}", p) for n, p in head_fe.named_parameters()]
    for n, p in iterables:
        if not p.requires_grad:
            continue
        if p.grad is None or float(p.grad.abs().sum().item()) == 0.0:
            no_grad_named.append((n, p.numel()))
        else:
            yes_grad_named.append((n, p.numel()))
    backbone.zero_grad(set_to_none=True); head_trans.zero_grad(set_to_none=True)
    if head_fe is not None:
        head_fe.zero_grad(set_to_none=True)
    backbone.train(bb_was); head_trans.train(ht_was)
    if head_fe is not None and hf_was is not None:
        head_fe.train(hf_was)
    return {
        "trainable_with_grad": len(yes_grad_named),
        "trainable_without_grad": len(no_grad_named),
        "trainable_without_grad_params": sum(n for _, n in no_grad_named),
        "no_grad_names_sample": [n for n, _ in no_grad_named[:8]],
    }


# ────────────────────────────────────────────────────────────────────────────
#  Per-component norms for wandb logging
# ────────────────────────────────────────────────────────────────────────────

def per_component_norms(groups: List[dict]) -> Dict[str, float]:
    """Return ``{name}/grad_norm`` and ``{name}/param_norm`` for each group."""
    out: Dict[str, float] = {}
    for g in groups:
        name = g["name"]
        ps = g["params"]
        gn2 = 0.0
        pn2 = 0.0
        for p in ps:
            if p.grad is not None:
                gn2 += float(p.grad.detach().pow(2).sum().item())
            pn2 += float(p.detach().pow(2).sum().item())
        out[f"{name}/grad_norm"] = gn2 ** 0.5
        out[f"{name}/param_norm"] = pn2 ** 0.5
        out[f"{name}/lr"] = float(g["lr"])
    return out


__all__ = [
    "detect_ckpt_flavor",
    "load_fe_from_ckpt",
    "load_layer_norm_from_ckpt",
    "load_post_extract_proj_from_ckpt",
    "load_transformer_from_ckpt",
    "load_head_from_ckpt",
    "freeze_unused_d2v_submodules",
    "apply_freezes",
    "build_param_groups",
    "apply_cosine_to_groups",
    "component_param_summary",
    "print_component_summary",
    "audit_gradient_flow",
    "per_component_norms",
]
