"""
Reconstruction training and analysis for SpectralFM.

  - train --recon_path fe:          FE + LayerNorm + MirrorDecoder (ConvTranspose1d), optional FE init from ``--ckpt``.
  - train --recon_path transformer: **End-to-end** ``Data2VecAudioModel`` (FE + transformer) + head mirroring the FE
                                      path (stem ``C→512``, ``LayerNorm(512)``, ``MirrorDecoder``). Init via ``--ckpt``.
                                      ``--freeze_fe`` freezes conv FE + Fairseq extractor ``layer_norm`` and sets
                                      ``feature_grad_mult=0``; still trains ``post_extract_proj``, encoder, head.
                                      ``--fe_ckpt`` optionally reloads FE+LN from a second Fairseq ``.pt`` after ``--ckpt``.
  - analyze:                        FE: ``--ckpt_ae`` only. Transformer: if checkpoint contains ``data2vec_audio``,
                                      no ``--ckpt`` needed; else legacy ``--ckpt`` or embedded ``backbone`` (v1).
  - interp:                         Pretrained interp decoder (full fairseq model)

Usage:
    python code/train_reconstruction.py --mode train --recon_path fe --lr 1e-4 --n_samples 1000 --steps 10000
    python code/train_reconstruction.py --mode train --recon_path transformer --ckpt none --n_samples 1000 --steps 2000
    # Default: --batch_size 512 --grad_accum_steps 4 (micro-batch 128 per forward; effective 512 per optimizer step)
    python code/train_reconstruction.py --mode train --recon_path transformer --ckpt /path/to/checkpoint_best.pt ...
    # Libri FE frozen + random transformer init: --ckpt none --fe_ckpt fairseq/base_libri_official.pt --freeze_fe
    python code/train_reconstruction.py --mode analyze --recon_path transformer --ckpt_ae path/to/ckpt_tr_....pt
    # add --ckpt fairseq.pt only for older checkpoints without ``data2vec_audio``
"""
import sys, os, warnings, logging, math, torch, argparse, json
from typing import List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from importlib import util as _ilu

# Per-component init/freeze/audit/LR helpers (sibling module under code/)
import recon_components as rc  # type: ignore  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("torio").setLevel(logging.ERROR)
logging.getLogger("numexpr").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

# ── Project path setup (same pattern as eval_fe_decoder.py) ──
_THIS_DIR = Path(__file__).resolve().parent          # code/
_ROOT = _THIS_DIR.parent                              # SpectralFM/
_FAIRSEQ_PATH = _ROOT / "fairseq"
for _p in [str(_THIS_DIR), str(_FAIRSEQ_PATH), str(_FAIRSEQ_PATH / "examples")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# FE conv spec: [(out_ch, kernel, stride)] — matches spectralfm yaml configs
_FE_CONV_LAYERS = [(512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 5, 5)]


def build_fe_standalone(device, ckpt_path=None):
    """Build ConvFeatureExtractionModel + LayerNorm directly, no full model needed.

    If ckpt_path points to a data2vec_audio checkpoint, loads FE weights from it.
    If ckpt_path is None or 'none', returns randomly-initialized FE.
    """
    from fairseq.models.wav2vec.wav2vec2 import ConvFeatureExtractionModel

    encoder = ConvFeatureExtractionModel(
        conv_layers=_FE_CONV_LAYERS, dropout=0.0,
        mode="layer_norm", conv_bias=False,
    )
    layer_norm = nn.LayerNorm(512)

    if ckpt_path and ckpt_path.lower() != "none" and os.path.isfile(ckpt_path):
        # Use the shared flat-keys reader so apr28-style nested ``encoder``/``layer_norm``
        # dicts also resolve (alongside fairseq audio and data2vec_multi layouts).
        sd = rc._read_state_dict(ckpt_path)
        fe_sub, src_pref = rc._fe_subdict(sd)
        if fe_sub:
            tgt_enc = encoder.state_dict()
            fe_ok = {k: v for k, v in fe_sub.items()
                     if k in tgt_enc and v.shape == tgt_enc[k].shape}
            if len(fe_ok) < len(fe_sub):
                print(f"[!] FE init: skipped {len(fe_sub) - len(fe_ok)} tensors (shape mismatch vs SpectralFM FE)")
            if fe_ok:
                encoder.load_state_dict(fe_ok, strict=False)
                print(f"[+] Loaded FE weights from {ckpt_path} via '{src_pref}' ({len(fe_ok)} tensors applied)")
            else:
                print(f"[!] No compatible FE tensors in {ckpt_path} — using random init")
        else:
            print(f"[!] No FE keys recognised in {ckpt_path} — using random init")
        ln_sub, ln_src = rc._ln_subdict(sd)
        if ln_sub:
            tgt_ln = layer_norm.state_dict()
            ln_ok = {k: v for k, v in ln_sub.items()
                     if k in tgt_ln and v.shape == tgt_ln[k].shape}
            if ln_ok:
                layer_norm.load_state_dict(ln_ok, strict=False)
                print(f"[+] Loaded LayerNorm from {ckpt_path} via '{ln_src}' ({len(ln_ok)} tensors applied)")
    else:
        print("[+] FE random init (no checkpoint)")

    return encoder.to(device), layer_norm.to(device)


def _load_full_model(ckpt_path, device):
    """Fallback: load via fairseq model_loader (needed for interp mode)."""
    _spec = _ilu.spec_from_file_location("model_loader", str(_THIS_DIR / "model_loader.py"))
    _ml = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_ml)
    return _ml.load_fairseq_checkpoint(ckpt_path, device=device)

# ─── Mirror decoder: reverse of FE conv layers ───
# Encoder (FE):  [(512,3,1)]×4 + [(512,5,5)]
#   Conv1d(1→512, k3,s1)   245→243
#   Conv1d(512→512, k3,s1) 243→241
#   Conv1d(512→512, k3,s1) 241→239
#   Conv1d(512→512, k3,s1) 239→237
#   Conv1d(512→512, k5,s5) 237→47
#
# Decoder (mirror):
#   ConvT(512→512, k5,s5,op2) 47→237
#   ConvT(512→512, k3,s1)     237→239
#   ConvT(512→512, k3,s1)     239→241
#   ConvT(512→512, k3,s1)     241→243
#   ConvT(512→1,   k3,s1)     243→245

class MirrorDecoder(nn.Module):
    """ConvTranspose1d decoder mirroring the FE's 5-layer CNN in reverse."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(512, 512, kernel_size=5, stride=5, output_padding=2),
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

    def forward(self, x):
        """x: [B, 512, 47] → [B, 1, 245]"""
        intermediates = []
        for layer in self.layers:
            x = layer(x)
            intermediates.append(x.detach().clone())
        return x, intermediates


class TransformerMirrorDecoder(nn.Module):
    """Map transformer encoder output ``[B, T, C]`` → spectrogram ``[B, 1, 245]``.

    Mirrors the FE reconstruction head layout: **1×1 conv ``C→512``** → **``LayerNorm(512)`` over time** (optional
    for checkpoints from the older frozen-decoder format) → **``MirrorDecoder``** (same upsampling as the FE path).
    ``T`` must match the subsampled FE length (47 for the default 245-bin conv stack).
    """

    def __init__(
        self,
        encoder_embed_dim: int = 768,
        mid_channels: int = 512,
        *,
        use_pre_decoder_ln: bool = True,
    ):
        super().__init__()
        self.encoder_embed_dim = int(encoder_embed_dim)
        self.mid_channels = int(mid_channels)
        self.stem = nn.Conv1d(self.encoder_embed_dim, self.mid_channels, kernel_size=1, bias=True)
        self.pre_decoder_ln = (
            nn.LayerNorm(self.mid_channels) if use_pre_decoder_ln else None
        )
        self.decoder = MirrorDecoder()

    def forward(self, x_bt_c):
        """``x_bt_c``: ``[B, T, C]`` from ``Data2VecAudioModel`` ``features_only`` encoder output."""
        x = x_bt_c.transpose(1, 2).contiguous()
        x = self.stem(x)
        if self.pre_decoder_ln is not None:
            x = self.pre_decoder_ln(x.transpose(1, 2)).transpose(1, 2)
        return self.decoder(x)


def _spectralfm_data2vec_audio_cfg():
    """OmegaConf for ``Data2VecAudioModel`` matching SpectralFM FE conv stack (245 bins → 47 steps)."""
    from omegaconf import OmegaConf, open_dict
    from data2vec.models.data2vec_audio import Data2VecAudioConfig

    sc = OmegaConf.structured(Data2VecAudioConfig)
    with open_dict(sc):
        sc.max_update = 21000
        sc.ema_anneal_end_step = 21000
        sc.conv_feature_layers = (
            "[(512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 5, 5)]"
        )
        sc.skip_pretrained_weights = True
        sc.model_path = None
        sc.recon_only = False
        sc.train_only_fe = False
        sc.encoder_layerdrop = 0.0
        sc.dropout_input = 0.0
        sc.dropout_features = 0.0
        sc.extractor_mode = "layer_norm"
        sc.encoder_embed_dim = 768
    return sc


def build_data2vec_audio_backbone(device, ckpt_path=None):
    """Build ``Data2VecAudioModel`` (fixed SpectralFM cfg), optionally merge weights like Fairseq ``build_model``.

    Always constructs the **audio** architecture from YAML-equivalent config, then loads tensors from
    ``ckpt_path`` via ``load_checkpoint_to_cpu`` + ``load_state_dict(..., strict=False)``, plus an explicit
    ``feature_extractor.*`` / ``layer_norm.*`` pass (same idea as ``Data2VecAudioModel.build_model`` when
    ``model_path`` is set). This matches Hydra training, which never restores a ``data2vec_multi`` graph from
    ``base_libri_official.pt`` — it only copies compatible keys into ``data2vec_audio``.

    Args:
        device: Torch device.
        ckpt_path: Fairseq ``.pt`` with ``state["model"]`` dict, or ``None`` / ``"none"`` for random init.

    Returns:
        ``Data2VecAudioModel`` on ``device``.
    """
    from fairseq import checkpoint_utils
    from data2vec.models.data2vec_audio import Data2VecAudioModel

    cfg = _spectralfm_data2vec_audio_cfg()
    model = Data2VecAudioModel(cfg).to(device)

    ck = (ckpt_path or "").strip()
    if not ck or ck.lower() == "none":
        print("[+] Data2VecAudio backbone: random init (no checkpoint)")
        return model

    if not os.path.isfile(ck):
        raise FileNotFoundError(f"checkpoint not found: {ck}")

    state = checkpoint_utils.load_checkpoint_to_cpu(ck, arg_overrides={})
    raw = state.get("model", state)
    if not isinstance(raw, dict):
        raw = {}
    raw.pop("_ema", None)

    fe_sd = {
        k[len("feature_extractor.") :]: v
        for k, v in raw.items()
        if k.startswith("feature_extractor.")
    }
    fe_src = "feature_extractor.*"
    if not fe_sd:
        # data2vec_multi / base_libri_official: conv stack lives under AUDIO modality encoder
        _multi_fe = "modality_encoders.AUDIO.local_encoder."
        fe_sd = {
            k[len(_multi_fe) :]: v
            for k, v in raw.items()
            if k.startswith(_multi_fe)
        }
        if fe_sd:
            fe_src = "modality_encoders.AUDIO.local_encoder.* → feature_extractor"
    if fe_sd:
        tgt = model.feature_extractor.state_dict()
        # Shape-safe merge: base_libri (data2vec_multi) uses a different conv spec than SpectralFM;
        # strict=False still errors on same-key shape mismatch, so drop incompatible tensors.
        compatible = {
            k: v
            for k, v in fe_sd.items()
            if k in tgt and v.shape == tgt[k].shape
        }
        n_skip = len(fe_sd) - len(compatible)
        if n_skip:
            print(
                f"[!] Skipped {n_skip} FE checkpoint tensors (shape mismatch vs SpectralFM FE; "
                f"e.g. multi AudioEncoder vs 245-bin conv stack)"
            )
        if compatible:
            fe_incomp = model.feature_extractor.load_state_dict(compatible, strict=False)
            n_fe = len(tgt)
            n_ok = n_fe - len(fe_incomp.missing_keys)
            print(
                f"[+] Merged feature_extractor from {fe_src}: {len(compatible)} tensors applied, "
                f"{n_ok}/{n_fe} submodule keys matched "
                f"({len(fe_incomp.unexpected_keys)} unexpected)"
            )
        elif fe_sd:
            print("[!] No compatible FE tensors to load (all shapes differ); FE stays at init")

    ln_sd = {
        k[len("layer_norm.") :]: v
        for k, v in raw.items()
        if k.startswith("layer_norm.")
    }
    if ln_sd:
        model.layer_norm.load_state_dict(ln_sd, strict=False)
        print(f"[+] Merged layer_norm.* ({len(ln_sd)} tensors from checkpoint)")

    full_incomp = model.load_state_dict(raw, strict=False)
    print(
        f"[+] Full-model strict=False merge: {len(full_incomp.missing_keys)} missing keys, "
        f"{len(full_incomp.unexpected_keys)} unexpected"
    )
    print(f"[+] Data2VecAudio backbone weights sourced from {ck}")
    return model


def merge_feature_extractor_and_extractor_ln_from_pt(model, ckpt_path: str) -> None:
    """Merge only ``feature_extractor.*`` and ``layer_norm.*`` (512-d post-conv LN) from a Fairseq ``.pt``.

    Used for ``--fe_ckpt`` after ``build_data2vec_audio_backbone`` so the conv FE (+ its LayerNorm) can come from a
    different checkpoint than the main ``--ckpt`` (e.g. Libri FE + random transformer init).
    """
    from fairseq import checkpoint_utils

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"--fe_ckpt not found: {ckpt_path}")
    state = checkpoint_utils.load_checkpoint_to_cpu(ckpt_path, arg_overrides={})
    raw = state.get("model", state)
    if not isinstance(raw, dict):
        print(f"[!] fe_ckpt has no model dict: {ckpt_path}")
        return
    raw.pop("_ema", None)

    fe_sd = {
        k[len("feature_extractor.") :]: v
        for k, v in raw.items()
        if k.startswith("feature_extractor.")
    }
    fe_src = "feature_extractor.*"
    if not fe_sd:
        _multi_fe = "modality_encoders.AUDIO.local_encoder."
        fe_sd = {
            k[len(_multi_fe) :]: v
            for k, v in raw.items()
            if k.startswith(_multi_fe)
        }
        if fe_sd:
            fe_src = "modality_encoders.AUDIO.local_encoder.* → feature_extractor"
    if fe_sd:
        tgt = model.feature_extractor.state_dict()
        compatible = {k: v for k, v in fe_sd.items() if k in tgt and v.shape == tgt[k].shape}
        n_skip = len(fe_sd) - len(compatible)
        if n_skip:
            print(
                f"[!] fe_ckpt: skipped {n_skip} FE tensors (shape mismatch vs SpectralFM FE)"
            )
        if compatible:
            model.feature_extractor.load_state_dict(compatible, strict=False)
            print(f"[+] fe_ckpt merged feature_extractor from {fe_src} ({len(compatible)} tensors)")
        else:
            print("[!] fe_ckpt: no compatible FE tensors")

    ln_sd = {
        k[len("layer_norm.") :]: v
        for k, v in raw.items()
        if k.startswith("layer_norm.")
    }
    if ln_sd:
        model.layer_norm.load_state_dict(ln_sd, strict=False)
        print(f"[+] fe_ckpt merged layer_norm.* ({len(ln_sd)} tensors)")


def _freeze_data2vec_conv_fe_and_extractor_ln(model) -> int:
    """Freeze conv FE + Fairseq ``layer_norm`` on extractor dim; return frozen param count."""
    n = 0
    for p in model.feature_extractor.parameters():
        p.requires_grad = False
        n += p.numel()
    for p in model.layer_norm.parameters():
        p.requires_grad = False
        n += p.numel()
    return n


def build_random_data2vec_audio(device):
    """Randomly initialized ``Data2VecAudioModel`` (SpectralFM 245-bin conv stack, no base checkpoint)."""
    return build_data2vec_audio_backbone(device, None)


def _freeze_module_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def load_data(data_dir, n_samples, device):
    import torchaudio
    wavs = sorted(Path(data_dir).glob("*.wav"))[:n_samples]
    inputs = []
    for w in wavs:
        x, sr = torchaudio.load(str(w))
        inputs.append(x.squeeze(0))
    source = torch.stack(inputs).to(device)
    return source, wavs


class LazyWavDataset(torch.utils.data.Dataset):
    """Reads from a fairseq-style manifest TSV (root_dir on line 1, then fname\\tsize)."""

    def __init__(self, manifest_path, max_samples=None, normalize=False):
        with open(manifest_path) as f:
            self.root = f.readline().strip()
            lines = f.readlines()
        if max_samples:
            lines = lines[:max_samples]
        self.fnames = []
        for ln in lines:
            parts = ln.strip().split("\t")
            self.fnames.append(parts[0])
        self.normalize = normalize

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        import soundfile as sf
        wav, _ = sf.read(os.path.join(self.root, self.fnames[idx]), dtype="float32")
        x = torch.from_numpy(wav).float()
        if self.normalize:
            x = F.layer_norm(x, x.shape)   # zero-mean unit-std, matches data2vec training
        return x


def exp_tag(args):
    rp = getattr(args, "recon_path", "fe")
    prefix = "tr_" if rp == "transformer" else ""
    base = f"{prefix}lr{args.lr}_n{args.n_samples}_s{args.steps}"
    if args.warmup > 0:
        base += f"_w{args.warmup}"
    if rp == "transformer" and getattr(args, "freeze_fe", False):
        base += "_freezeFE"
    suf = getattr(args, "run_suffix", None)
    if suf:
        base += f"_{str(suf).strip().replace(' ', '_')}"
    return base


def _wandb_display_name(args, tag: str) -> str:
    return (getattr(args, "wandb_run_name", None) or "").strip() or tag


def cosine_lr(step, total_steps, warmup_steps, peak_lr):
    """Cosine schedule: linear warmup → cosine decay to 0."""
    if step <= warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _tv_loss(x: "torch.Tensor") -> "torch.Tensor":
    """Mean absolute first-order difference along the time axis (L1-TV).
    x: [B, T]  →  scalar.  TV = mean_B mean_{t=1..T-1} |x_{t} - x_{t-1}|
    """
    return (x[:, 1:] - x[:, :-1]).abs().mean()


def _configure_single_gpu(device: str) -> None:
    """Pin the default CUDA device for single-GPU training (no DataParallel / DDP)."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    rest = str(device)[len("cuda") :]
    idx = int(rest[1:]) if rest.startswith(":") else 0
    if idx < 0 or idx >= torch.cuda.device_count():
        raise ValueError(
            f"--device {device!r} invalid: {torch.cuda.device_count()} CUDA device(s) visible"
        )
    torch.cuda.set_device(idx)
    if torch.cuda.device_count() > 1:
        print(
            f"[i] Single-GPU: default CUDA device = {idx} "
            f"({torch.cuda.get_device_name(idx)}); "
            f"{torch.cuda.device_count()} GPU(s) visible — use CUDA_VISIBLE_DEVICES to hide others."
        )


def resolve_micro_batch_accum(n, batch_size, grad_accum_steps):
    """Effective batch per optimizer step = micro_batch * accum (<= batch_size, capped by n).

    Requires ``batch_size % grad_accum_steps == 0`` so micro-batch is integral.
    """
    ga = max(1, int(grad_accum_steps))
    bs = int(batch_size)
    if bs % ga != 0:
        raise ValueError(
            f"batch_size ({bs}) must be divisible by grad_accum_steps ({ga}); "
            f"e.g. 512 and 4 → micro-batch 128"
        )
    micro = bs // ga
    if n < micro:
        return n, 1, n
    # Largest effective batch <= min(bs, n) using full micro-batches only
    max_full = (min(bs, n) // micro) * micro
    if max_full == 0:
        return n, 1, n
    if max_full < micro * ga:
        ga_adj = max(1, max_full // micro)
        return micro, ga_adj, micro * ga_adj
    return micro, ga, bs


# ═══════════════════════════════════════════════════════
#  MODE: train — train autoencoder, save checkpoint
# ═══════════════════════════════════════════════════════

def _build_save_obj_3ae(
    *,
    backbone, head_trans, head_fe, head_proj,
    args, tag, freeze_map,
    lambda_trans, lambda_fe, lambda_proj,
    monitor_fe, monitor_proj,
    fe_extra, enc_dim, rows, init_manifest,
    lr_overrides, audit,
    losses, losses_trans, losses_fe, losses_proj, lrs,
    ckpt_label, micro_bs, eff_bs,
    step: int, completed: bool,
    normalize: bool = False,
):
    """Build the canonical save_obj dict for the 3-AE transformer trainer.

    Used by both the final end-of-training save and the periodic rolling
    ``ckpt_<tag>_last.pt`` save (``--save_every``). ``step`` is the number of
    optimizer steps completed so far (i.e. ``len(losses)`` at call time);
    ``completed`` flags whether this is the final save (True) or a rolling
    intermediate (False).
    """
    so = {
        "recon_path": "transformer",
        "train_reconstruction_format": 5,
        "transformer_ae_format": 3,
        "normalize": normalize,
        "freeze_map": freeze_map,
        "lambda_recon_trans": lambda_trans,
        "lambda_recon_fe":    lambda_fe,
        "lambda_recon_proj":  lambda_proj,
        "head_fe_enabled":    head_fe is not None,
        "head_proj_enabled":  head_proj is not None,
        "monitor_recon_fe":   monitor_fe,
        "monitor_recon_proj": monitor_proj,
        "fe_ckpt": fe_extra or None,
        "feature_grad_mult_at_save": float(getattr(backbone, "feature_grad_mult", 1.0)),
        "encoder_embed_dim": enc_dim,
        "data2vec_audio": backbone.state_dict(),
        "transformer_mirror": head_trans.state_dict(),
        "components": rows,
        "init_manifest": init_manifest,
        "lr_overrides": {k: float(v) for k, v in lr_overrides.items() if v is not None},
        "audit_gradient_flow": audit,
        "losses": list(losses),
        "losses_trans": list(losses_trans),
        "losses_fe":    list(losses_fe),
        "losses_proj":  list(losses_proj),
        "lrs": list(lrs),
        "tag": tag,
        "lr": args.lr,
        "warmup": args.warmup,
        "n_samples": args.n_samples,
        "steps": args.steps,
        "step_saved": int(step),
        "completed": bool(completed),
        "backbone_ckpt": ckpt_label,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "micro_batch_size": micro_bs,
        "effective_batch_size": eff_bs,
        "random_init_proj":   bool(getattr(args, "random_init_proj", False)),
    }
    if head_fe is not None:
        so["fe_mirror"] = head_fe.state_dict()
    if head_proj is not None:
        so["proj_mirror"] = head_proj.state_dict()
    return so


def _run_train_transformer_autoencoder(args):
    """Train ``Data2VecAudioModel`` + reconstruction head (MSE on first 245 samples).

    Per-component setup is driven by ``--init_{fe,ln,proj,transformer,head}_ckpt`` and the
    matching ``--freeze_*`` and ``--lr_*`` flags (default LR = ``--lr`` for all groups).
    Back-compat: ``--ckpt`` seeds the full backbone via the old fairseq merge, ``--fe_ckpt``
    additionally re-merges FE+LN, and ``--freeze_fe`` freezes the conv FE + extractor LN.

    Forward uses ``features_only=True, mask=False`` so ``mask_emb``, ``final_proj``,
    ``fe_recon_decoder``, ``trans_recon_decoder``, and EMA never receive gradients;
    those are explicitly frozen and reported by the audit step below.
    """
    tag = exp_tag(args)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1) Build backbone (legacy --ckpt path keeps current Feb-25 / random behavior) ──
    ckpt_arg = (args.ckpt or "").strip()
    if not ckpt_arg or ckpt_arg.lower() == "none":
        backbone = build_data2vec_audio_backbone(args.device, None)
        ckpt_label = "random_backbone"
    else:
        if not os.path.isfile(ckpt_arg):
            raise FileNotFoundError(f"--ckpt not found: {ckpt_arg}")
        backbone = build_data2vec_audio_backbone(args.device, ckpt_arg)
        ckpt_label = ckpt_arg

    fe_extra = (getattr(args, "fe_ckpt", None) or "").strip()
    if fe_extra:
        merge_feature_extractor_and_extractor_ln_from_pt(backbone, fe_extra)

    enc_dim = int(backbone.cfg.encoder_embed_dim)
    head_trans = TransformerMirrorDecoder(enc_dim, use_pre_decoder_ln=True).to(args.device)

    # ── 2a) Resolve init paths (back-compat: --init_head_ckpt -> --init_head_trans_ckpt) ──
    init_head_trans_path = (
        (getattr(args, "init_head_trans_ckpt", None) or "").strip()
        or (getattr(args, "init_head_ckpt", None) or "").strip()
    )
    init_head_fe_path   = (getattr(args, "init_head_fe_ckpt",   None) or "").strip()
    init_head_proj_path = (getattr(args, "init_head_proj_ckpt", None) or "").strip()

    # ── 2b) Decide if head_fe / head_proj are needed (Option B: enable when user opts in) ──
    user_set_lambda_fe    = getattr(args, "lambda_recon_fe",    None) is not None
    user_set_lambda_trans = getattr(args, "lambda_recon_trans", None) is not None
    user_set_lambda_proj  = getattr(args, "lambda_recon_proj",  None) is not None
    want_head_fe = (
        bool(init_head_fe_path and init_head_fe_path.lower() != "none")
        or (user_set_lambda_fe and float(args.lambda_recon_fe) > 0)
        or bool(getattr(args, "monitor_recon_fe", False))
    )
    want_head_proj = (
        bool(init_head_proj_path and init_head_proj_path.lower() != "none")
        or (user_set_lambda_proj and float(args.lambda_recon_proj) > 0)
        or bool(getattr(args, "monitor_recon_proj", False))
    )
    head_fe   = MirrorDecoder().to(args.device) if want_head_fe   else None
    head_proj = (
        TransformerMirrorDecoder(enc_dim, use_pre_decoder_ln=True).to(args.device)
        if want_head_proj else None
    )

    # ── 2c) Per-component init from explicit ``--init_*_ckpt`` overrides ──
    init_manifest: dict = {}
    print("\n[init] Per-component pretrained-weight loading:")
    init_specs: List[Tuple[str, str, object]] = [
        ("fe",          init_fe := (getattr(args, "init_fe_ckpt", None) or "").strip(),          (rc.load_fe_from_ckpt,                backbone)),
        ("ln",          init_ln := (getattr(args, "init_ln_ckpt", None) or "").strip(),          (rc.load_layer_norm_from_ckpt,        backbone)),
        ("proj",        init_pj := (getattr(args, "init_proj_ckpt", None) or "").strip(),        (rc.load_post_extract_proj_from_ckpt, backbone)),
        ("transformer", init_tr := (getattr(args, "init_transformer_ckpt", None) or "").strip(), (rc.load_transformer_from_ckpt,       backbone)),
        ("head_trans",  init_head_trans_path,                                                    (rc.load_head_from_ckpt,              head_trans)),
    ]
    if head_fe is not None:
        init_specs.append(("head_fe", init_head_fe_path, (rc.load_head_from_ckpt, head_fe)))
    if head_proj is not None:
        init_specs.append(("head_proj", init_head_proj_path, (rc.load_head_from_ckpt, head_proj)))
    for name, path, (loader, target) in init_specs:
        if not path or path.lower() == "none":
            continue
        init_manifest[name] = loader(target, path)

    # ── 2d) Optional: random-init post_extract_proj AFTER all per-component loads,
    #       so we can wipe a proj that came in via --ckpt / --init_proj_ckpt.  Useful
    #       for the 3-AE setup where the transformer comes from a pretrained ckpt
    #       (e.g. May-18 + base_libri overlay) but the proj must train from scratch.
    if bool(getattr(args, "random_init_proj", False)):
        proj_mod = getattr(backbone, "post_extract_proj", None)
        if proj_mod is None:
            print("[init] --random_init_proj: backbone.post_extract_proj is None (skipping)")
        else:
            n_reset = 0
            for sub in proj_mod.modules():
                if hasattr(sub, "reset_parameters") and callable(sub.reset_parameters):
                    sub.reset_parameters()
                    n_reset += 1
            print(f"[init] --random_init_proj: re-initialized post_extract_proj "
                  f"({type(proj_mod).__name__}, reset {n_reset} submodules)")
            init_manifest["proj"] = {"source": "random_init_proj=True", "loaded": 0}

    # ── 3) Apply freezes (per-component + always-freeze unused submodules) ──
    freeze_unused, unused_names = rc.freeze_unused_d2v_submodules(backbone)
    if freeze_unused:
        print(f"\n[freeze-unused] Froze {freeze_unused:,} params in unused submodules: {unused_names}")
    freeze_map = {
        "fe":          bool(getattr(args, "freeze_fe", False) or getattr(args, "freeze_fe_v2", False)),
        "ln":          bool(getattr(args, "freeze_ln", False)),
        "proj":        bool(getattr(args, "freeze_proj", False)),
        "transformer": bool(getattr(args, "freeze_transformer", False)),
        "head_trans":  bool(getattr(args, "freeze_head_trans", False) or getattr(args, "freeze_head", False)),
        "head_fe":     bool(getattr(args, "freeze_head_fe", False)),
        "head_proj":   bool(getattr(args, "freeze_head_proj", False)),
    }
    frozen_comp = rc.apply_freezes(backbone, head_trans, head_fe, head_proj, freeze=freeze_map)
    if frozen_comp:
        print(f"[freeze-comp] {frozen_comp}")
    if freeze_map["fe"]:
        backbone.feature_grad_mult = 0.0
        print("[freeze-comp] fe: also set feature_grad_mult=0 (no-grad conv forward).")

    backbone.train(); head_trans.train()
    if head_fe is not None:
        head_fe.train()
    if head_proj is not None:
        head_proj.train()

    # ── 3b) Resolve λ_fe / λ_trans / λ_proj with Option B defaults + auto-zero rules ──
    if user_set_lambda_trans:
        lambda_trans = float(args.lambda_recon_trans)
    else:
        lambda_trans = 1.0  # Option B default for --recon_path transformer
    if user_set_lambda_fe:
        lambda_fe = float(args.lambda_recon_fe)
    else:
        lambda_fe = 1.0 if head_fe is not None else 0.0
    if user_set_lambda_proj:
        lambda_proj = float(args.lambda_recon_proj)
    else:
        lambda_proj = 1.0 if head_proj is not None else 0.0
    lambda_tv_fe = float(getattr(args, "lambda_tv_fe", 0.0))

    # Auto-zero rules (with warnings). A frozen head still contributes 0 gradient,
    # but we may want to keep the *forward* alive for monitoring; that's gated by
    # --monitor_recon_{fe,proj} below.
    if lambda_trans > 0 and freeze_map["head_trans"]:
        print(f"[lambda] head_trans is frozen → forcing lambda_recon_trans {lambda_trans} → 0")
        lambda_trans = 0.0
    if lambda_fe > 0 and (head_fe is None or freeze_map["head_fe"]):
        reason = "head_fe not built" if head_fe is None else "head_fe is frozen"
        print(f"[lambda] {reason} → forcing lambda_recon_fe {lambda_fe} → 0")
        lambda_fe = 0.0
    if lambda_proj > 0 and (head_proj is None or freeze_map["head_proj"]):
        reason = "head_proj not built" if head_proj is None else "head_proj is frozen"
        print(f"[lambda] {reason} → forcing lambda_recon_proj {lambda_proj} → 0")
        lambda_proj = 0.0
    # Monitoring flags: when set AND λ is 0 (frozen / disabled), still run the head
    # forward under no_grad to log the loss value for evaluation.
    monitor_fe   = bool(getattr(args, "monitor_recon_fe",   False)) and head_fe   is not None
    monitor_proj = bool(getattr(args, "monitor_recon_proj", False)) and head_proj is not None
    if lambda_fe == 0.0 and lambda_trans == 0.0 and lambda_proj == 0.0:
        raise RuntimeError(
            "All recon lambdas (trans / fe / proj) resolved to 0 — no trainable reconstruction loss. "
            "Set --lambda_recon_{trans,fe,proj}>0 and ensure the matching head is not frozen."
        )
    print(
        f"[lambda] λ_recon_trans={lambda_trans}  λ_recon_fe={lambda_fe}  λ_recon_proj={lambda_proj}  "
        f"(head_fe={'on' if head_fe is not None else 'off'}, "
        f"head_proj={'on' if head_proj is not None else 'off'}, "
        f"monitor_fe={monitor_fe}, monitor_proj={monitor_proj})"
    )

    # ── 4) Component summary + dry-run gradient audit ──
    rows = rc.component_param_summary(backbone, head_trans, head_fe, head_proj)
    rc.print_component_summary(rows, tag=f"trainable components [{tag}]")
    bb_total = sum(p.numel() for p in backbone.parameters())
    bb_train = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    hd_total = sum(p.numel() for p in head_trans.parameters()) + (
        sum(p.numel() for p in head_fe.parameters()) if head_fe is not None else 0
    ) + (sum(p.numel() for p in head_proj.parameters()) if head_proj is not None else 0)
    hd_train = sum(p.numel() for p in head_trans.parameters() if p.requires_grad) + (
        sum(p.numel() for p in head_fe.parameters() if p.requires_grad) if head_fe is not None else 0
    ) + (sum(p.numel() for p in head_proj.parameters() if p.requires_grad) if head_proj is not None else 0)

    # ── 5) Per-component param groups with optional per-component LR ──
    lr_overrides = {
        "fe":          getattr(args, "lr_fe", None),
        "ln":          getattr(args, "lr_ln", None),
        "proj":        getattr(args, "lr_proj", None),
        "transformer": getattr(args, "lr_transformer", None),
        "head_trans":  getattr(args, "lr_head_trans", None) if getattr(args, "lr_head_trans", None) is not None
                       else getattr(args, "lr_head", None),
        "head_fe":     getattr(args, "lr_head_fe", None),
        "head_proj":   getattr(args, "lr_head_proj", None),
    }
    param_groups = rc.build_param_groups(
        backbone, head_trans, head_fe, head_proj,
        lr_base=float(args.lr), lr_overrides=lr_overrides,
    )
    train_mode_str = "frozen_fe_train_encoder_and_head" if freeze_map["fe"] else "full_data2vec_audio_ae"
    print(f"\n{'='*60}")
    print(f"TRAIN transformer AE  [{tag}]  mode={train_mode_str}")
    print(f"Optimizer param groups ({len(param_groups)}):")
    for g in param_groups:
        n_p = sum(p.numel() for p in g["params"])
        print(f"  - {g['name']:<12} params={n_p:>12,}  base_lr={g['base_lr']:.2e}")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(param_groups, lr=args.lr)
    use_cosine = args.warmup > 0
    if use_cosine:
        print(f"LR schedule: cosine (warmup={args.warmup}, decay to 0; applied per-group)")

    use_lazy = args.manifest is not None
    normalize = getattr(args, "normalize", False)
    if use_lazy:
        ds = LazyWavDataset(args.manifest, max_samples=args.n_samples, normalize=normalize)
        n = len(ds)
        micro_bs, accum_steps, eff_bs = resolve_micro_batch_accum(
            n, args.batch_size, args.grad_accum_steps
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=micro_bs, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=(n >= micro_bs * accum_steps),
        )
        loader_iter = iter(loader)
        print(f"Lazy loader: {n} samples from {args.manifest}")
    else:
        source_all, _ = load_data(args.data_dir, args.n_samples, args.device)
        target_all = source_all[:, :245].float()
        n = source_all.shape[0]
        micro_bs, accum_steps, eff_bs = resolve_micro_batch_accum(
            n, args.batch_size, args.grad_accum_steps
        )

    print(
        f"Samples: {n}  micro_batch={micro_bs}  grad_accum={accum_steps}  "
        f"effective_batch={eff_bs}  optimizer_steps: {args.steps}  LR: {args.lr}\n"
    )
    if eff_bs != args.batch_size:
        print(
            f"[!] effective_batch {eff_bs} < requested {args.batch_size} "
            f"(dataset or n too small for 512 with accum {args.grad_accum_steps})\n"
        )

    # ── 6) Gradient-flow audit (one tiny dry forward+backward) ──
    if use_lazy:
        try:
            audit_src = next(iter(loader)).to(args.device)
        except StopIteration:
            audit_src = None
    else:
        audit_src = source_all[: min(4, n)]
    audit = None
    if audit_src is not None and audit_src.numel() > 0:
        audit = rc.audit_gradient_flow(
            backbone, head_trans, audit_src,
            micro_bs=min(4, audit_src.shape[0]),
            head_fe=head_fe, head_proj=head_proj,
            lambda_trans=lambda_trans, lambda_fe=lambda_fe, lambda_proj=lambda_proj,
        )
        print(
            f"\n[grad-audit] trainable_with_grad={audit['trainable_with_grad']}  "
            f"trainable_without_grad={audit['trainable_without_grad']} "
            f"({audit['trainable_without_grad_params']:,} params)  "
            f"sample-no-grad-names={audit['no_grad_names_sample']}"
        )

    wb_run = None
    if args.wandb_project:
        import wandb
        wb_cfg = {
            "lr": args.lr, "warmup": args.warmup,
            "n_samples": n, "steps": args.steps,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "micro_batch_size": micro_bs,
            "effective_batch_size": eff_bs,
            "loss_fn": "L2 (MSE)",
            "schedule": "cosine" if use_cosine else "constant",
            "backbone_params_total": bb_total,
            "backbone_params_trainable": bb_train,
            "head_params_total": hd_total,
            "head_params_trainable": hd_train,
            "recon_path": "transformer",
            "transformer_train_mode": train_mode_str,
            "freeze_fe": freeze_map["fe"],
            "freeze_ln": freeze_map["ln"],
            "freeze_proj": freeze_map["proj"],
            "freeze_transformer": freeze_map["transformer"],
            "freeze_head_trans": freeze_map["head_trans"],
            "freeze_head_fe":    freeze_map["head_fe"],
            "freeze_head_proj":  freeze_map["head_proj"],
            "lambda_recon_trans": lambda_trans,
            "lambda_recon_fe":    lambda_fe,
            "lambda_recon_proj":  lambda_proj,
            "head_fe_enabled":    head_fe is not None,
            "head_proj_enabled":  head_proj is not None,
            "monitor_recon_fe":   monitor_fe,
            "monitor_recon_proj": monitor_proj,
            "random_init_proj":   bool(getattr(args, "random_init_proj", False)),
            "fe_ckpt": fe_extra or None,
            "train_reconstruction_format": 3,
            "backbone_ckpt": ckpt_label,
            "encoder_embed_dim": enc_dim,
            "manifest": getattr(args, "manifest", None),
            "data_dir": getattr(args, "data_dir", None),
            "exp_tag": tag,
            "components": rows,
            "init_manifest": init_manifest,
            "lr_overrides": {k: float(v) for k, v in lr_overrides.items() if v is not None},
            "param_groups": [{"name": g["name"], "base_lr": g["base_lr"],
                              "n_params": sum(p.numel() for p in g["params"])} for g in param_groups],
            "audit_gradient_flow": audit,
        }
        wb_run = wandb.init(project=args.wandb_project,
                            name=_wandb_display_name(args, tag), config=wb_cfg)

    log_interval = max(1, args.steps // 20)
    losses, lrs = [], []

    def _next_lazy_batch():
        nonlocal loader_iter
        try:
            b = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            b = next(loader_iter)
        return b.to(args.device)

    # Forward hooks: capture post-LN FE for head_fe and post-extract-proj for head_proj.
    # Registered only when the corresponding head is enabled.
    _captured: dict = {}
    def _ln_hook(_mod, _inp, out):
        _captured["fe_seq"] = out
    def _proj_hook(_mod, _inp, out):
        _captured["proj_seq"] = out
    _ln_handle   = backbone.layer_norm.register_forward_hook(_ln_hook)   if head_fe   is not None else None
    _proj_handle = (
        backbone.post_extract_proj.register_forward_hook(_proj_hook)
        if head_proj is not None else None
    )

    losses_trans: list = []
    losses_fe:    list = []
    losses_proj:  list = []

    for step in range(1, args.steps + 1):
        lrs_now = rc.apply_cosine_to_groups(
            param_groups, step, args.steps, args.warmup if use_cosine else 0
        )
        # Track the representative (max) LR for the legacy ``losses/lrs`` columns
        lr_now = max(lrs_now.values()) if lrs_now else args.lr
        lrs.append(lr_now)

        optimizer.zero_grad()
        loss_total_sum = 0.0
        loss_trans_sum  = 0.0
        loss_fe_sum     = 0.0
        loss_fe_tv_sum  = 0.0
        loss_proj_sum   = 0.0
        target_var_sum = 0.0
        pred_trans_var_sum = 0.0
        pred_fe_var_sum    = 0.0
        pred_proj_var_sum  = 0.0
        for _ in range(accum_steps):
            if use_lazy:
                source_batch = _next_lazy_batch()
            else:
                idx = torch.randint(0, n, (micro_bs,), device=args.device)
                source_batch = source_all[idx]
            target = source_batch[:, :245].float()
            backbone.train(); head_trans.train()
            if head_fe is not None:
                head_fe.train()
            if head_proj is not None:
                head_proj.train()
            _captured.clear()
            enc_out = backbone(source_batch, padding_mask=None, mask=False, features_only=True)

            micro_loss = torch.zeros((), device=source_batch.device)
            if lambda_trans > 0:
                recon_t, _ = head_trans(enc_out["x"])
                recon_t = recon_t.squeeze(1)
                lt = F.mse_loss(recon_t, target)
                micro_loss = micro_loss + lambda_trans * lt
                loss_trans_sum += float(lt.item())
                pred_trans_var_sum += float(recon_t.var(dim=-1).mean().item())
            if lambda_fe > 0 and head_fe is not None:
                fe_seq = _captured.get("fe_seq")
                if fe_seq is None:
                    raise RuntimeError("fe_seq hook did not fire — backbone.layer_norm forward path changed?")
                # fe_seq is [B, T=47, C=512]; MirrorDecoder wants [B, C, T].
                recon_f, _ = head_fe(fe_seq.transpose(1, 2).contiguous())
                recon_f = recon_f.squeeze(1)
                lf_mse = F.mse_loss(recon_f, target)
                lf_tv  = _tv_loss(recon_f) if lambda_tv_fe > 0.0 else 0.0
                lf     = lf_mse + lambda_tv_fe * lf_tv
                micro_loss = micro_loss + lambda_fe * lf
                loss_fe_sum += float(lf_mse.item())
                loss_fe_tv_sum += float(lf_tv.item() if lambda_tv_fe > 0.0 else 0.0)
                pred_fe_var_sum += float(recon_f.var(dim=-1).mean().item())
            elif monitor_fe:
                # Forward-only FE recon for logging when head_fe is frozen / λ=0.
                fe_seq = _captured.get("fe_seq")
                if fe_seq is not None:
                    with torch.no_grad():
                        recon_f, _ = head_fe(fe_seq.transpose(1, 2).contiguous())
                        recon_f = recon_f.squeeze(1)
                        lf = F.mse_loss(recon_f, target)
                    loss_fe_sum += float(lf.item())
                    pred_fe_var_sum += float(recon_f.var(dim=-1).mean().item())
            if lambda_proj > 0 and head_proj is not None:
                proj_seq = _captured.get("proj_seq")
                if proj_seq is None:
                    raise RuntimeError("proj_seq hook did not fire — backbone.post_extract_proj forward path changed?")
                # proj_seq is [B, T=47, C=enc_dim]; TransformerMirrorDecoder consumes [B, T, C].
                recon_p, _ = head_proj(proj_seq)
                recon_p = recon_p.squeeze(1)
                lp = F.mse_loss(recon_p, target)
                micro_loss = micro_loss + lambda_proj * lp
                loss_proj_sum += float(lp.item())
                pred_proj_var_sum += float(recon_p.var(dim=-1).mean().item())
            elif monitor_proj:
                proj_seq = _captured.get("proj_seq")
                if proj_seq is not None:
                    with torch.no_grad():
                        recon_p, _ = head_proj(proj_seq)
                        recon_p = recon_p.squeeze(1)
                        lp = F.mse_loss(recon_p, target)
                    loss_proj_sum += float(lp.item())
                    pred_proj_var_sum += float(recon_p.var(dim=-1).mean().item())
            target_var_sum += float(target.var(dim=-1).mean().item())
            (micro_loss / accum_steps).backward()
            loss_total_sum += float(micro_loss.item())
        optimizer.step()
        # accum-averaged scalars
        loss_total_avg = loss_total_sum / accum_steps
        loss_trans_avg = loss_trans_sum / accum_steps if lambda_trans > 0 else 0.0
        loss_fe_avg    = (loss_fe_sum    / accum_steps) if ((lambda_fe   > 0 and head_fe   is not None) or monitor_fe) else 0.0
        loss_fe_tv_avg = (loss_fe_tv_sum / accum_steps) if (lambda_fe   > 0 and lambda_tv_fe > 0.0)                    else 0.0
        loss_proj_avg  = (loss_proj_sum / accum_steps) if ((lambda_proj > 0 and head_proj is not None) or monitor_proj) else 0.0
        target_var_avg = target_var_sum / accum_steps
        pred_trans_var_avg = pred_trans_var_sum / accum_steps if lambda_trans > 0 else 0.0
        pred_fe_var_avg    = (pred_fe_var_sum   / accum_steps) if ((lambda_fe   > 0 and head_fe   is not None) or monitor_fe)   else 0.0
        pred_proj_var_avg  = (pred_proj_var_sum / accum_steps) if ((lambda_proj > 0 and head_proj is not None) or monitor_proj) else 0.0
        losses.append(loss_total_avg)
        losses_trans.append(loss_trans_avg)
        losses_fe.append(loss_fe_avg)
        losses_proj.append(loss_proj_avg)

        epoch = step * eff_bs / n
        if wb_run:
            log_d = {
                "train/loss_total":     loss_total_avg,
                "train/loss":           loss_total_avg,  # back-compat key
                "train/lambda_trans":   lambda_trans,
                "train/lambda_fe":      lambda_fe,
                "train/lambda_proj":    lambda_proj,
                "train/target_var":     target_var_avg,
                "train/lr":             lr_now,
                "train/epoch":          epoch,
                "step":                 step,
            }
            if lambda_trans > 0:
                log_d["train/recon_trans_loss"] = loss_trans_avg
                log_d["train/pred_var_trans"]   = pred_trans_var_avg
            if (lambda_fe > 0 and head_fe is not None) or monitor_fe:
                log_d["train/recon_fe_loss"]    = loss_fe_avg
                log_d["train/pred_var_fe"]      = pred_fe_var_avg
                if monitor_fe and lambda_fe == 0:
                    log_d["train/recon_fe_loss_monitor"] = loss_fe_avg
                if lambda_tv_fe > 0.0:
                    log_d["train/recon_fe_tv_loss"] = loss_fe_tv_avg
                    log_d["train/lambda_tv_fe"]     = lambda_tv_fe
            if (lambda_proj > 0 and head_proj is not None) or monitor_proj:
                log_d["train/recon_proj_loss"]  = loss_proj_avg
                log_d["train/pred_var_proj"]    = pred_proj_var_avg
                if monitor_proj and lambda_proj == 0:
                    log_d["train/recon_proj_loss_monitor"] = loss_proj_avg
            log_d.update({f"train/lr/{k}": v for k, v in lrs_now.items()})
            if step % 100 == 0 or step == 1:
                log_d.update(rc.per_component_norms(param_groups))
                log_d["train/grad_norm"] = sum(
                    v ** 2 for k, v in log_d.items() if k.endswith("/grad_norm")
                ) ** 0.5
            wb_run.log(log_d, step=step)

        if step % log_interval == 0 or step == 1:
            parts = [f"L2_total = {loss_total_avg:.6f}"]
            if lambda_trans > 0:
                parts.append(f"L2_trans = {loss_trans_avg:.6f}")
            if (lambda_fe > 0 and head_fe is not None) or monitor_fe:
                tag_fe = "L2_fe(mon)" if (monitor_fe and lambda_fe == 0) else "L2_fe"
                parts.append(f"{tag_fe} = {loss_fe_avg:.6f}")
                if lambda_tv_fe > 0.0:
                    parts.append(f"TV_fe = {loss_fe_tv_avg:.6f}")
            if (lambda_proj > 0 and head_proj is not None) or monitor_proj:
                tag_pj = "L2_proj(mon)" if (monitor_proj and lambda_proj == 0) else "L2_proj"
                parts.append(f"{tag_pj} = {loss_proj_avg:.6f}")
            parts.append(f"lr = {lr_now:.2e}  epoch = {epoch:.1f}")
            print(f"  step {step:5d}/{args.steps}  " + "  ".join(parts))

        save_every = int(getattr(args, "save_every", 0) or 0)
        if save_every > 0 and step % save_every == 0 and step < args.steps:
            last_path = str(Path(args.out_dir) / f"ckpt_{tag}_last.pt")
            tmp_path  = last_path + ".tmp"
            save_obj_rolling = _build_save_obj_3ae(
                backbone=backbone, head_trans=head_trans, head_fe=head_fe, head_proj=head_proj,
                args=args, tag=tag, freeze_map=freeze_map,
                lambda_trans=lambda_trans, lambda_fe=lambda_fe, lambda_proj=lambda_proj,
                monitor_fe=monitor_fe, monitor_proj=monitor_proj,
                fe_extra=fe_extra, enc_dim=enc_dim, rows=rows, init_manifest=init_manifest,
                lr_overrides=lr_overrides, audit=audit,
                losses=losses, losses_trans=losses_trans, losses_fe=losses_fe, losses_proj=losses_proj,
                lrs=lrs, ckpt_label=ckpt_label, micro_bs=micro_bs, eff_bs=eff_bs,
                step=step, completed=False, normalize=normalize,
            )
            torch.save(save_obj_rolling, tmp_path)
            os.replace(tmp_path, last_path)
            print(f"  [save] step {step:5d}/{args.steps} -> {last_path}")
            if wb_run:
                wb_run.log({"ckpt/last_step": step, "ckpt/last_path": last_path}, step=step)

    if _ln_handle is not None:
        _ln_handle.remove()
    if _proj_handle is not None:
        _proj_handle.remove()

    ckpt_path = str(Path(args.out_dir) / f"ckpt_{tag}.pt")
    save_obj = _build_save_obj_3ae(
        backbone=backbone, head_trans=head_trans, head_fe=head_fe, head_proj=head_proj,
        args=args, tag=tag, freeze_map=freeze_map,
        lambda_trans=lambda_trans, lambda_fe=lambda_fe, lambda_proj=lambda_proj,
        monitor_fe=monitor_fe, monitor_proj=monitor_proj,
        fe_extra=fe_extra, enc_dim=enc_dim, rows=rows, init_manifest=init_manifest,
        lr_overrides=lr_overrides, audit=audit,
        losses=losses, losses_trans=losses_trans, losses_fe=losses_fe, losses_proj=losses_proj,
        lrs=lrs, ckpt_label=ckpt_label, micro_bs=micro_bs, eff_bs=eff_bs,
        step=len(losses), completed=True, normalize=normalize,
    )
    torch.save(save_obj, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")
    print(f"Final train L2 (total): {losses[-1]:.6f}"
          + (f"  trans={losses_trans[-1]:.6f}" if lambda_trans > 0 else "")
          + (f"  fe={losses_fe[-1]:.6f}"       if ((lambda_fe   > 0 and head_fe   is not None) or monitor_fe)   else "")
          + (f"  proj={losses_proj[-1]:.6f}"   if ((lambda_proj > 0 and head_proj is not None) or monitor_proj) else ""))
    if wb_run:
        final_log = {"train/final_loss": losses[-1]}
        if lambda_trans > 0:
            final_log["train/final_recon_trans_loss"] = losses_trans[-1]
        if (lambda_fe > 0 and head_fe is not None) or monitor_fe:
            final_log["train/final_recon_fe_loss"] = losses_fe[-1]
        if (lambda_proj > 0 and head_proj is not None) or monitor_proj:
            final_log["train/final_recon_proj_loss"] = losses_proj[-1]
        wb_run.log(final_log)
        wb_run.finish()


def run_train_mode(args):
    if getattr(args, "recon_path", "fe") == "transformer":
        return _run_train_transformer_autoencoder(args)

    if getattr(args, "freeze_fe", False) or (getattr(args, "fe_ckpt", None) or "").strip():
        print("[!] --freeze_fe / --fe_ckpt are only used with --recon_path transformer (ignored here).")

    tag = exp_tag(args)
    os.makedirs(args.out_dir, exist_ok=True)

    encoder, layer_norm = build_fe_standalone(args.device, args.ckpt)
    decoder = MirrorDecoder().to(args.device)

    fe_init_manifest = {"ckpt": args.ckpt}

    ln_params = sum(p.numel() for p in layer_norm.parameters())
    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    fe_rows = [
        {"name": "fe",      "class": type(encoder).__name__,
         "total": enc_params, "trainable": enc_params, "frozen": 0},
        {"name": "ln",      "class": type(layer_norm).__name__,
         "total": ln_params,  "trainable": ln_params,  "frozen": 0},
        {"name": "decoder", "class": type(decoder).__name__,
         "total": dec_params, "trainable": dec_params, "frozen": 0},
    ]
    print(f"\n{'='*60}")
    print(f"TRAIN FE-AE  [{tag}]")
    rc.print_component_summary(fe_rows, tag=f"FE-AE components [{tag}]")
    print(f"{'='*60}")

    all_params = list(encoder.parameters()) + list(layer_norm.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr)

    use_cosine = args.warmup > 0
    if use_cosine:
        print(f"LR schedule: cosine (warmup={args.warmup}, peak={args.lr}, decay to 0)")

    use_lazy = args.manifest is not None
    if use_lazy:
        ds = LazyWavDataset(args.manifest, max_samples=args.n_samples)
        n = len(ds)
        micro_bs, accum_steps, eff_bs = resolve_micro_batch_accum(
            n, args.batch_size, args.grad_accum_steps
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=micro_bs, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=(n >= micro_bs * accum_steps),
        )
        loader_iter = iter(loader)
        print(f"Lazy loader: {n} samples from {args.manifest}")
    else:
        source_all, _ = load_data(args.data_dir, args.n_samples, args.device)
        target_all = source_all[:, :245].float()
        n = source_all.shape[0]
        micro_bs, accum_steps, eff_bs = resolve_micro_batch_accum(
            n, args.batch_size, args.grad_accum_steps
        )

    print(
        f"Samples: {n}  micro_batch={micro_bs}  grad_accum={accum_steps}  "
        f"effective_batch={eff_bs}  optimizer_steps: {args.steps}  LR: {args.lr}\n"
    )
    if eff_bs != args.batch_size:
        print(
            f"[!] effective_batch {eff_bs} < requested {args.batch_size} "
            f"(dataset or n too small)\n"
        )

    wb_run = None
    if args.wandb_project:
        import wandb
        wb_run = wandb.init(
            project=args.wandb_project,
            name=_wandb_display_name(args, tag),
            config={
                "lr": args.lr, "warmup": args.warmup,
                "n_samples": n, "steps": args.steps,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "micro_batch_size": micro_bs,
                "effective_batch_size": eff_bs,
                "loss_fn": "L2 (MSE)",
                "schedule": "cosine" if use_cosine else "constant",
                "enc_params": enc_params, "dec_params": dec_params, "ln_params": ln_params,
                "ckpt_init": args.ckpt or "random",
                "loader": "lazy" if use_lazy else "preload",
                "recon_path": "fe",
                "manifest": getattr(args, "manifest", None),
                "data_dir": getattr(args, "data_dir", None),
                "exp_tag": tag,
                "components": fe_rows,
                "init_manifest": fe_init_manifest,
            },
        )

    encoder.train()
    decoder.train()

    log_interval = max(1, args.steps // 20)
    losses = []
    lrs = []

    def _next_lazy_batch_fe():
        nonlocal loader_iter
        try:
            b = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            b = next(loader_iter)
        return b.to(args.device)

    for step in range(1, args.steps + 1):
        if use_cosine:
            lr_now = cosine_lr(step, args.steps, args.warmup, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
        else:
            lr_now = args.lr
        lrs.append(lr_now)

        optimizer.zero_grad()
        loss_sum = 0.0
        for _ in range(accum_steps):
            if use_lazy:
                source_batch = _next_lazy_batch_fe()
            else:
                idx = torch.randint(0, n, (micro_bs,), device=args.device)
                source_batch = source_all[idx]
            target = source_batch[:, :245].float()
            fe_out = encoder(source_batch)
            fe_normed = layer_norm(fe_out.transpose(1, 2)).transpose(1, 2)
            recon, _ = decoder(fe_normed)
            loss = F.mse_loss(recon.squeeze(1), target) / accum_steps
            loss.backward()
            loss_sum += loss.item() * accum_steps
        optimizer.step()
        losses.append(loss_sum)

        epoch = step * eff_bs / n

        if wb_run:
            log_d = {"train/loss": loss_sum, "train/lr": lr_now,
                     "train/epoch": epoch, "step": step}
            if step % 100 == 0 or step == 1:
                grad_norm = sum(p.grad.norm().item() ** 2
                                for p in all_params if p.grad is not None) ** 0.5
                log_d["train/grad_norm"] = grad_norm
            wb_run.log(log_d, step=step)

        if step % log_interval == 0 or step == 1:
            print(f"  step {step:5d}/{args.steps}  L2 = {loss_sum:.6f}  lr = {lr_now:.2e}  epoch = {epoch:.1f}")

    ckpt_path = str(Path(args.out_dir) / f"ckpt_{tag}.pt")
    torch.save({
        "encoder": encoder.state_dict(),
        "layer_norm": layer_norm.state_dict(),
        "decoder": decoder.state_dict(),
        "losses": losses,
        "lrs": lrs,
        "tag": tag,
        "lr": args.lr,
        "warmup": args.warmup,
        "n_samples": args.n_samples,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "micro_batch_size": micro_bs,
        "effective_batch_size": eff_bs,
        "components": fe_rows,
        "init_manifest": fe_init_manifest,
        "recon_path": "fe",
        "train_reconstruction_format": 3,
    }, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")
    print(f"Final train L2: {losses[-1]:.6f}")

    if wb_run:
        wb_run.log({"train/final_loss": losses[-1]})
        wb_run.finish()


# ═══════════════════════════════════════════════════════
#  MODE: analyze — load checkpoint, evaluate on datasets
# ═══════════════════════════════════════════════════════

_NOVA_DATASETS = {
    "single_channel_10k": "single_channel_10k/wav",
    "single_channel_1k":  "single_channel_1k/wavs",
    "single_channel_100": "single_channel_100/wavs",
    "multi_channel":      "multi_channel/wavs",
    "labeled_data":       "labeled_data/wavs",
}

def _resolve_nova_root():
    """Find the nova_data root on this machine."""
    for base in ["/mnt5/noy/SpectralFM", "/storage/noy/SpectralFM"]:
        p = os.path.join(base, "fairseq/data/nova_data")
        if os.path.isdir(p):
            return p
    return None


def _load_ae_model(ckpt_path, device):
    """Load encoder + decoder from an autoencoder checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder, layer_norm = build_fe_standalone(device, ckpt_path=None)
    decoder = MirrorDecoder().to(device)
    encoder.load_state_dict(ckpt["encoder"])
    layer_norm.load_state_dict(ckpt["layer_norm"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    return encoder, layer_norm, decoder, ckpt


def _eval_on_samples(encoder, layer_norm, decoder, source, device):
    """Run reconstruction, return per-sample L2 array and predictions."""
    target = source[:, :245].float()
    with torch.no_grad():
        fe_out = encoder(source)
        fe_normed = layer_norm(fe_out.transpose(1, 2)).transpose(1, 2)
        recon, intermediates = decoder(fe_normed)
        pred = recon.squeeze(1)
        per_sample_l2 = ((pred - target) ** 2).mean(dim=1)
    return per_sample_l2, pred, target, fe_out, fe_normed, intermediates


def _load_transformer_mirror_from_ckpt(ckpt_ae_path, device):
    """Load ``TransformerMirrorDecoder`` (head) from a train-mode transformer checkpoint."""
    ckpt = torch.load(ckpt_ae_path, map_location=device)
    enc_dim = int(ckpt.get("encoder_embed_dim", 768))
    ae_fmt = int(ckpt.get("transformer_ae_format", 1))
    use_ln = ae_fmt >= 2
    head = TransformerMirrorDecoder(enc_dim, use_pre_decoder_ln=use_ln).to(device)
    sd = ckpt.get("transformer_mirror", {})
    if sd:
        inc = head.load_state_dict(sd, strict=False)
        if inc.missing_keys or inc.unexpected_keys:
            print(
                f"[!] transformer_mirror load_state_dict(strict=False): "
                f"{len(inc.missing_keys)} missing, {len(inc.unexpected_keys)} unexpected"
            )
    head.eval()
    return head, ckpt


def _eval_transformer_mirror_on_samples(backbone, head, source, device):
    """Backbone + head; same return shape tuple as ``_eval_on_samples``."""
    target = source[:, :245].float()
    with torch.no_grad():
        backbone.eval()
        head.eval()
        out = backbone(source, padding_mask=None, mask=False, features_only=True)
        lat = out["x"]
        recon, intermediates = head(lat)
        pred = recon.squeeze(1)
        per_sample_l2 = ((pred - target) ** 2).mean(dim=1)
    fe_dummy = torch.zeros(source.shape[0], 512, 47, device=source.device, dtype=source.dtype)
    fe_norm_dummy = fe_dummy.transpose(1, 2)
    return per_sample_l2, pred, target, fe_dummy, fe_norm_dummy, intermediates


def _run_analyze_transformer_mirror(args):
    """Analyze datasets using ``Data2VecAudioModel`` + ``TransformerMirrorDecoder`` (eval, frozen)."""
    head, ckpt = _load_transformer_mirror_from_ckpt(args.ckpt_ae, args.device)
    tag = ckpt["tag"]

    d2v_sd = ckpt.get("data2vec_audio")
    emb = ckpt.get("backbone")
    ck = (args.ckpt or "").strip()

    if isinstance(d2v_sd, dict) and d2v_sd:
        backbone = build_data2vec_audio_backbone(args.device, None)
        inc = backbone.load_state_dict(d2v_sd, strict=False)
        if inc.missing_keys or inc.unexpected_keys:
            print(
                f"[!] data2vec_audio load_state_dict(strict=False): "
                f"{len(inc.missing_keys)} missing, {len(inc.unexpected_keys)} unexpected"
            )
        if "feature_grad_mult_at_save" in ckpt:
            backbone.feature_grad_mult = float(ckpt["feature_grad_mult_at_save"])
        _freeze_module_params(backbone)
        backbone_desc = "ckpt_ae[data2vec_audio]"
    elif isinstance(emb, dict) and emb:
        backbone = build_data2vec_audio_backbone(args.device, None)
        inc = backbone.load_state_dict(emb, strict=False)
        if inc.missing_keys or inc.unexpected_keys:
            print(
                f"[!] Embedded backbone load_state_dict(strict=False): "
                f"{len(inc.missing_keys)} missing, {len(inc.unexpected_keys)} unexpected"
            )
        _freeze_module_params(backbone)
        backbone_desc = "embedded legacy `backbone` in ckpt_ae"
    elif ckpt.get("backbone_ckpt") == "random_backbone":
        print(
            "ERROR: v1 random-backbone checkpoint (decoder-only, no full ``data2vec_audio``). "
            "Re-train with ``train_reconstruction_format`` 2 to save ``data2vec_audio`` for analyze."
        )
        sys.exit(1)
    elif ck and ck.lower() != "none" and os.path.isfile(ck):
        backbone = build_data2vec_audio_backbone(args.device, ck)
        _freeze_module_params(backbone)
        backbone_desc = ck
    else:
        print(
            "ERROR: transformer analyze needs one of:\n"
            "  - ``data2vec_audio`` in ckpt_ae (full-model AE checkpoints), or\n"
            "  - legacy embedded ``backbone`` (frozen random v1), or\n"
            "  - ``--ckpt`` fairseq .pt for v1 Libri-init decoder-only checkpoints."
        )
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"ANALYZE transformer AE  [{tag}]  head={args.ckpt_ae}  backbone={backbone_desc}")
    print(f"{'='*60}")
    saved_rows = ckpt.get("components")
    if isinstance(saved_rows, list) and saved_rows:
        rc.print_component_summary(saved_rows, tag=f"saved components [{tag}]")
    if isinstance(ckpt.get("init_manifest"), dict) and ckpt["init_manifest"]:
        print("\n  ── init_manifest (from training) ──")
        for k, v in ckpt["init_manifest"].items():
            print(f"    {k:<12} {v}")
    if isinstance(ckpt.get("lr_overrides"), dict) and ckpt["lr_overrides"]:
        print(f"\n  per-component LR overrides at train time: {ckpt['lr_overrides']}")

    nova_root = _resolve_nova_root()
    datasets_to_eval = args.datasets.split(",") if args.datasets else ["single_channel_10k"]
    n_stat = args.n_stat
    all_dataset_stats = {}

    for ds_name in datasets_to_eval:
        ds_name = ds_name.strip()
        if ds_name in _NOVA_DATASETS:
            wav_dir = os.path.join(nova_root, _NOVA_DATASETS[ds_name]) if nova_root else None
        else:
            wav_dir = ds_name if os.path.isdir(ds_name) else None

        if not wav_dir or not os.path.isdir(wav_dir):
            print(f"\n[!] Dataset '{ds_name}' not found at {wav_dir}, skipping")
            continue

        print(f"\n── Dataset: {ds_name} ({wav_dir}) ──")
        source_stat, wavs_stat = load_data(wav_dir, n_stat, args.device)
        per_l2, pred_stat, target_stat, _, _, _ = _eval_transformer_mirror_on_samples(
            backbone, head, source_stat, args.device)
        l2_arr = per_l2.cpu().numpy()

        stats = {
            "n": len(l2_arr),
            "mean": float(np.mean(l2_arr)),
            "std": float(np.std(l2_arr)),
            "median": float(np.median(l2_arr)),
            "min": float(np.min(l2_arr)),
            "max": float(np.max(l2_arr)),
            "p5": float(np.percentile(l2_arr, 5)),
            "p95": float(np.percentile(l2_arr, 95)),
        }
        all_dataset_stats[ds_name] = (stats, l2_arr)
        print(f"  N={stats['n']}  Mean L2={stats['mean']:.6f}  Std={stats['std']:.6f}")
        print(f"  Median={stats['median']:.6f}  Min={stats['min']:.6f}  Max={stats['max']:.6f}")
        print(f"  P5={stats['p5']:.6f}  P95={stats['p95']:.6f}")

        k = min(args.k, len(l2_arr))
        sorted_idx = np.argsort(l2_arr)
        best_idx = sorted_idx[:k]
        worst_idx = sorted_idx[-k:][::-1]
        print(f"  Best  {k}: indices {best_idx.tolist()}, L2={l2_arr[best_idx].tolist()}")
        print(f"  Worst {k}: indices {worst_idx.tolist()}, L2={l2_arr[worst_idx].tolist()}")
        plot_best_worst(target_stat, pred_stat, per_l2, wavs_stat,
                        best_idx, worst_idx, ds_name, tag, args.out_dir)

    if all_dataset_stats:
        plot_stats(all_dataset_stats, tag, args.out_dir)


def run_analyze_mode(args):
    os.makedirs(args.out_dir, exist_ok=True)
    if getattr(args, "recon_path", "fe") == "transformer":
        _run_analyze_transformer_mirror(args)
        return

    encoder, layer_norm, decoder, ckpt = _load_ae_model(args.ckpt_ae, args.device)
    tag = ckpt["tag"]
    losses = ckpt["losses"]

    print(f"\n{'='*60}")
    print(f"ANALYZE FE-AE  [{tag}]  from {args.ckpt_ae}")
    print(f"{'='*60}")
    saved_rows = ckpt.get("components")
    if isinstance(saved_rows, list) and saved_rows:
        rc.print_component_summary(saved_rows, tag=f"saved components [{tag}]")
    if isinstance(ckpt.get("init_manifest"), dict) and ckpt["init_manifest"]:
        print("\n  ── init_manifest (from training) ──")
        for k, v in ckpt["init_manifest"].items():
            print(f"    {k:<12} {v}")

    nova_root = _resolve_nova_root()
    datasets_to_eval = args.datasets.split(",") if args.datasets else ["single_channel_10k"]
    n_stat = args.n_stat

    all_dataset_stats = {}

    for ds_name in datasets_to_eval:
        ds_name = ds_name.strip()
        if ds_name in _NOVA_DATASETS:
            wav_dir = os.path.join(nova_root, _NOVA_DATASETS[ds_name]) if nova_root else None
        else:
            wav_dir = ds_name if os.path.isdir(ds_name) else None

        if not wav_dir or not os.path.isdir(wav_dir):
            print(f"\n[!] Dataset '{ds_name}' not found at {wav_dir}, skipping")
            continue

        print(f"\n── Dataset: {ds_name} ({wav_dir}) ──")

        source_stat, wavs_stat = load_data(wav_dir, n_stat, args.device)
        per_l2, pred_stat, target_stat, _, _, _ = _eval_on_samples(
            encoder, layer_norm, decoder, source_stat, args.device)
        l2_arr = per_l2.cpu().numpy()

        stats = {
            "n": len(l2_arr),
            "mean": float(np.mean(l2_arr)),
            "std": float(np.std(l2_arr)),
            "median": float(np.median(l2_arr)),
            "min": float(np.min(l2_arr)),
            "max": float(np.max(l2_arr)),
            "p5": float(np.percentile(l2_arr, 5)),
            "p95": float(np.percentile(l2_arr, 95)),
        }
        all_dataset_stats[ds_name] = (stats, l2_arr)

        print(f"  N={stats['n']}  Mean L2={stats['mean']:.6f}  Std={stats['std']:.6f}")
        print(f"  Median={stats['median']:.6f}  Min={stats['min']:.6f}  Max={stats['max']:.6f}")
        print(f"  P5={stats['p5']:.6f}  P95={stats['p95']:.6f}")

        k = min(args.k, len(l2_arr))
        sorted_idx = np.argsort(l2_arr)
        best_idx = sorted_idx[:k]
        worst_idx = sorted_idx[-k:][::-1]

        print(f"  Best  {k}: indices {best_idx.tolist()}, L2={l2_arr[best_idx].tolist()}")
        print(f"  Worst {k}: indices {worst_idx.tolist()}, L2={l2_arr[worst_idx].tolist()}")

        plot_best_worst(target_stat, pred_stat, per_l2, wavs_stat,
                        best_idx, worst_idx, ds_name, tag, args.out_dir)

    if all_dataset_stats:
        plot_stats(all_dataset_stats, tag, args.out_dir)


SAMPLE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12",
                 "#1abc9c", "#e67e22", "#34495e"]


def plot_best_worst(target, pred, per_l2, wavs, best_idx, worst_idx,
                    ds_name, tag, out_dir):
    k = len(best_idx)
    fig, axes = plt.subplots(k, 4, figsize=(24, 3.5 * k))
    if k == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"[{tag}] {ds_name} — Best {k} vs Worst {k}", fontsize=14, y=1.005)

    l2_np = per_l2.cpu().numpy()
    target_np = target.cpu().numpy()
    pred_np = pred.cpu().numpy()

    for row in range(k):
        bi, wi = int(best_idx[row]), int(worst_idx[row])

        ax = axes[row, 0]
        ax.plot(target_np[bi], color="#2c3e50", linewidth=1.2, label="GT")
        ax.plot(pred_np[bi], color="#2ecc71", linewidth=1.2, linestyle="--", label="Pred")
        fname_b = wavs[bi].name if hasattr(wavs[bi], "name") else str(bi)
        ax.set_title(f"Best #{row+1} — L2={l2_np[bi]:.5f}  ({fname_b})", fontsize=9)
        ax.legend(fontsize=7)
        if row == 0:
            ax.set_xlabel("")
        ax.set_ylabel("Amplitude")

        ax = axes[row, 1]
        res_b = pred_np[bi] - target_np[bi]
        ax.bar(range(len(res_b)), res_b,
               color=np.where(res_b >= 0, "#2ecc71", "#27ae60"), width=1.0, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"Best #{row+1} residual", fontsize=9)

        ax = axes[row, 2]
        ax.plot(target_np[wi], color="#2c3e50", linewidth=1.2, label="GT")
        ax.plot(pred_np[wi], color="#e74c3c", linewidth=1.2, linestyle="--", label="Pred")
        fname_w = wavs[wi].name if hasattr(wavs[wi], "name") else str(wi)
        ax.set_title(f"Worst #{row+1} — L2={l2_np[wi]:.5f}  ({fname_w})", fontsize=9)
        ax.legend(fontsize=7)

        ax = axes[row, 3]
        res_w = pred_np[wi] - target_np[wi]
        ax.bar(range(len(res_w)), res_w,
               color=np.where(res_w >= 0, "#e74c3c", "#c0392b"), width=1.0, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"Worst #{row+1} residual", fontsize=9)

    plt.tight_layout()
    safe_ds = ds_name.replace("/", "_")
    out_path = str(Path(out_dir) / f"bestworst_{safe_ds}_{tag}.png")
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Best/Worst plot saved to {out_path}")


def plot_stats(all_dataset_stats, tag, out_dir):
    n_ds = len(all_dataset_stats)
    fig, axes = plt.subplots(1, n_ds + 1, figsize=(6 * (n_ds + 1), 5))
    if n_ds + 1 == 1:
        axes = [axes]
    fig.suptitle(f"Reconstruction Stats [{tag}]", fontsize=14, y=1.01)

    ds_names = list(all_dataset_stats.keys())
    means = [all_dataset_stats[d][0]["mean"] for d in ds_names]
    stds = [all_dataset_stats[d][0]["std"] for d in ds_names]

    for i, ds_name in enumerate(ds_names):
        stats, l2_arr = all_dataset_stats[ds_name]
        ax = axes[i]
        ax.hist(l2_arr, bins=30, color=SAMPLE_COLORS[i % len(SAMPLE_COLORS)],
                edgecolor="white", alpha=0.85)
        ax.axvline(stats["mean"], color="black", linestyle="--", linewidth=1.5,
                   label=f"mean={stats['mean']:.4f}")
        ax.axvline(stats["median"], color="gray", linestyle=":", linewidth=1.5,
                   label=f"median={stats['median']:.4f}")
        ax.set_title(f"{ds_name} (n={stats['n']})")
        ax.set_xlabel("Per-sample L2")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    ax = axes[-1]
    x = np.arange(len(ds_names))
    ax.bar(x, means, yerr=stds, color=[SAMPLE_COLORS[i % len(SAMPLE_COLORS)]
           for i in range(len(ds_names))], edgecolor="white", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean L2")
    ax.set_title("Cross-dataset comparison")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.001, f"{m:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    out_path = str(Path(out_dir) / f"stats_{tag}.png")
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Stats plot saved to {out_path}")


def plot_analysis(source, fe_out, fe_normed, intermediates,
                  pred, target, losses, wavs, final_loss, per_sample_l2,
                  tag, out_dir, k):
    fig, axes = plt.subplots(k + 2, 2, figsize=(18, 4 * (k + 2)))
    fig.suptitle(f"Autoencoder [{tag}] — eval L2={final_loss.item():.4f}", fontsize=14, y=0.995)

    ax = axes[0, 0]
    ax.plot(losses, color="#e74c3c", linewidth=0.8)
    ax.set_title("Training L2 loss curve")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")

    ax = axes[0, 1]
    im = ax.imshow(fe_out[0].cpu().numpy(), aspect="auto", cmap="viridis")
    ax.set_title("Encoder output sample 0 [512, 47]")
    ax.set_xlabel("Time")
    ax.set_ylabel("Channel")
    plt.colorbar(im, ax=ax, fraction=0.046)

    for i in range(k):
        c = SAMPLE_COLORS[i % len(SAMPLE_COLORS)]
        gt = target[i].cpu().numpy()
        pr = pred[i].cpu().numpy()
        res = pr - gt
        l2 = per_sample_l2[i].item()

        ax = axes[i + 1, 0]
        ax.plot(gt, color="#2c3e50", linewidth=1.2, label="GT")
        ax.plot(pr, color=c, linewidth=1.2, linestyle="--", label="Pred")
        ax.set_title(f"Sample {i} — L2={l2:.4f}  ({wavs[i].name})")
        ax.set_xlabel("Bin")
        ax.legend(fontsize=8)

        ax = axes[i + 1, 1]
        ax.bar(range(245), res,
               color=np.where(res >= 0, "#3498db", "#e74c3c"), width=1.0)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"Sample {i} residual (pred − GT)")
        ax.set_xlabel("Bin")

    ax = axes[k + 1, 0]
    for i in range(k):
        c = SAMPLE_COLORS[i % len(SAMPLE_COLORS)]
        ax.plot(target[i].cpu().numpy(), color=c, linewidth=0.8, alpha=0.4)
        ax.plot(pred[i].cpu().numpy(), color=c, linewidth=1.2, linestyle="--", label=f"s{i}")
    ax.set_title(f"All {k} samples: thin=GT, dashed=pred (same color per sample)")
    ax.set_xlabel("Bin")
    ax.legend(fontsize=7, ncol=k)

    ax = axes[k + 1, 1]
    im = ax.imshow(intermediates[0][0].cpu().numpy(), aspect="auto", cmap="viridis")
    ax.set_title("Decoder step 1: ConvT(k5,s5) → [512, 237]")
    ax.set_xlabel("Time")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    fname = f"analyze_{tag}.png"
    out_path = str(Path(out_dir) / fname)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


# ═══════════════════════════════════════════════════════
#  MODE: interp — inference with pretrained interp decoder
# ═══════════════════════════════════════════════════════

def run_interp_mode(args):
    model, model_cfg, _ = _load_full_model(args.ckpt, device=args.device)
    model.eval()
    source, wavs = load_data(args.data_dir, args.n_samples, args.device)
    with torch.no_grad():
        features = model.feature_extractor(source)
        step1 = features.clone()
        features = model.layer_norm(features.transpose(1, 2))
        fe_seq = features
        decoder = model.fe_recon_decoder
        d4 = fe_seq.transpose(1, 2)
        d5 = F.interpolate(d4, size=decoder.out_dim, mode="linear", align_corners=False)
        d7 = decoder.fc(d5.transpose(1, 2)).squeeze(-1)
        recon_target = source[:, :245].float()
        loss = F.l1_loss(d7, recon_target)
        print(f"L1 loss: {loss.item():.6f}")
    out_path = str(_ROOT / "debug_recon_interp_steps.png")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(source[0].cpu().numpy())
    axes[0, 0].set_title("Input [245]")
    im = axes[0, 1].imshow(step1[0].cpu().numpy(), aspect="auto", cmap="viridis")
    axes[0, 1].set_title("FE output [512, 47]")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
    axes[1, 0].plot(recon_target[0].cpu().numpy(), label="GT")
    axes[1, 0].plot(d7[0].cpu().numpy(), "--", label="Pred")
    axes[1, 0].set_title(f"Recon (L1={loss.item():.4f})")
    axes[1, 0].legend()
    res = (d7[0] - recon_target[0]).cpu().numpy()
    axes[1, 1].bar(range(245), res, color=np.where(res >= 0, "#3498db", "#e74c3c"), width=1)
    axes[1, 1].set_title("Residual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruction training and evaluation for SpectralFM FE")
    parser.add_argument("--mode", choices=["train", "analyze", "interp"], default="train")
    parser.add_argument(
        "--recon_path",
        choices=["fe", "transformer"],
        default="fe",
        help="fe: CNN FE + LayerNorm + MirrorDecoder. transformer: Data2VecAudio + head; see --freeze_fe / --fe_ckpt.",
    )
    parser.add_argument("--ckpt", default="checkpoints/runai/recon_only_l1/20k/recon_interp_lr1e-4_20k.pt",
                        help="fe train: FE weights from data2vec ckpt or 'none'. "
                             "transformer train/analyze: weight dict merged into SpectralFM Data2VecAudio "
                             "(Fairseq-style; e.g. base_libri FE keys + random transformer where no match) "
                             "or 'none' for random backbone. "
                             "interp mode: still uses ensemble checkpoint restore via model_loader.")
    parser.add_argument("--ckpt_ae", default=None, help="Autoencoder checkpoint (for analyze mode)")
    parser.add_argument("--data_dir", default="fairseq/data/nova_data/single_channel_10k/wav")
    parser.add_argument("--manifest", default=None,
                        help="Path to fairseq manifest TSV (lazy loading, overrides --data_dir)")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Target effective batch size per optimizer step (must be divisible by --grad_accum_steps).",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation micro-steps (micro-batch = batch_size / this).",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=0,
                        help="Linear warmup steps for cosine LR schedule (0=constant LR)")
    parser.add_argument(
        "--save_every", type=int, default=0,
        help="Save a rolling ckpt_<tag>_last.pt to --out_dir every N optimizer steps "
             "(0 = only the final ckpt at end of training). Currently honored by the "
             "transformer-AE training path (--recon_path transformer).",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of samples to plot in analyze mode")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated dataset names for analyze mode "
                             "(e.g. single_channel_10k,multi_channel,labeled_data)")
    parser.add_argument("--n_stat", type=int, default=100,
                        help="Number of samples for statistical evaluation per dataset")
    parser.add_argument("--out_dir", default="autoencoder_experiments")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply F.layer_norm per sample at load time (zero-mean unit-std, "
                             "matching data2vec training preprocessing)")
    parser.add_argument("--wandb_project", default=None, help="W&B project name (omit to disable)")
    parser.add_argument(
        "--wandb_run_name",
        default=None,
        help="W&B run display name (default: exp tag from lr/n/steps/warmup/run_suffix)",
    )
    parser.add_argument(
        "--run_suffix",
        default=None,
        help="Appended to checkpoint exp tag (e.g. random vs pretrained_base_libri) so runs do not collide",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Single-GPU only: cuda or cuda:N (pins default device; no multi-GPU).",
    )
    parser.add_argument(
        "--freeze_fe",
        action="store_true",
        help="Transformer path only: freeze conv feature_extractor (alias for --freeze_fe_v2 + freeze_ln). "
        "Sets feature_grad_mult=0; trains post_extract_proj, TransformerEncoder, and recon head.",
    )
    parser.add_argument(
        "--fe_ckpt",
        default=None,
        help="Transformer path only: optional Fairseq .pt merged into feature_extractor + extractor layer_norm "
        "after the main --ckpt build (e.g. Libri FE weights with --ckpt none for random transformer).",
    )

    # ── Per-component pretrained-weight loading (transformer path) ──
    g_init = parser.add_argument_group(
        "Per-component init",
        "Each --init_*_ckpt loads ONLY that component's weights from the given file, "
        "auto-detecting fairseq_audio / data2vec_multi / apr28_fe_recon layouts and "
        "shape-mismatched tensors are skipped with a count. Applied AFTER --ckpt / --fe_ckpt.",
    )
    g_init.add_argument("--init_fe_ckpt",          default=None, help="conv feature_extractor source")
    g_init.add_argument("--init_ln_ckpt",          default=None, help="post-FE LayerNorm (512-d) source")
    g_init.add_argument("--init_proj_ckpt",        default=None, help="post_extract_proj (512->encoder_dim) source")
    g_init.add_argument("--init_transformer_ckpt", default=None,
                        help="TransformerEncoder source (fairseq encoder.* or data2vec_multi blocks.* w/ ViT remap)")
    g_init.add_argument("--init_head_ckpt",        default=None,
                        help="DEPRECATED alias for --init_head_trans_ckpt (kept for back-compat).")
    g_init.add_argument("--init_head_trans_ckpt",  default=None,
                        help="TransformerMirrorDecoder (head_trans) source: transformer_mirror or FE-AE decoder dict.")
    g_init.add_argument("--init_head_fe_ckpt",     default=None,
                        help="MirrorDecoder (head_fe) source: FE-AE decoder dict or transformer_mirror.decoder.")
    g_init.add_argument("--init_head_proj_ckpt",   default=None,
                        help="TransformerMirrorDecoder (head_proj) source: transformer_mirror or "
                             "FE-AE decoder dict. head_proj reads post-extract-proj features (B,T,enc_dim).")
    g_init.add_argument("--random_init_proj", action="store_true",
                        help="After all --ckpt / --init_*_ckpt loads, reset backbone.post_extract_proj to "
                             "random init. Useful when --ckpt brings in a trained proj you want to discard "
                             "(e.g. 3-AE: keep transformer + FE/LN from ckpt, but proj must train from scratch).")

    # ── Per-component freezing (transformer path) ──
    g_fr = parser.add_argument_group("Per-component freeze")
    g_fr.add_argument("--freeze_fe_v2",       action="store_true", help="freeze conv FE only (no LN side-effect)")
    g_fr.add_argument("--freeze_ln",          action="store_true", help="freeze post-FE LayerNorm")
    g_fr.add_argument("--freeze_proj",        action="store_true", help="freeze post_extract_proj")
    g_fr.add_argument("--freeze_transformer", action="store_true", help="freeze TransformerEncoder")
    g_fr.add_argument("--freeze_head",        action="store_true",
                      help="DEPRECATED alias for --freeze_head_trans (kept for back-compat).")
    g_fr.add_argument("--freeze_head_trans",  action="store_true",
                      help="freeze transformer-output reconstruction head (head_trans).")
    g_fr.add_argument("--freeze_head_fe",     action="store_true",
                      help="freeze post-LN FE reconstruction head (head_fe). Auto-zeroes lambda_recon_fe.")
    g_fr.add_argument("--freeze_head_proj",   action="store_true",
                      help="freeze post-extract-proj reconstruction head (head_proj). Auto-zeroes lambda_recon_proj.")

    # ── Per-component LR overrides (transformer path) ──
    g_lr = parser.add_argument_group(
        "Per-component LR",
        "Each --lr_<name> overrides the base --lr for that component's Adam param group. "
        "Cosine warmup/decay is applied per-group using each group's base_lr.",
    )
    g_lr.add_argument("--lr_fe",          type=float, default=None)
    g_lr.add_argument("--lr_ln",          type=float, default=None)
    g_lr.add_argument("--lr_proj",        type=float, default=None)
    g_lr.add_argument("--lr_transformer", type=float, default=None)
    g_lr.add_argument("--lr_head",        type=float, default=None,
                      help="DEPRECATED alias for --lr_head_trans (kept for back-compat).")
    g_lr.add_argument("--lr_head_trans",  type=float, default=None)
    g_lr.add_argument("--lr_head_fe",     type=float, default=None)
    g_lr.add_argument("--lr_head_proj",   type=float, default=None)

    # ── Reconstruction loss weights (transformer path; FE path always uses recon_fe) ──
    g_loss = parser.add_argument_group(
        "Reconstruction loss weights",
        "Option B defaults: with --recon_path transformer and no explicit lambdas, λ_trans=1.0 and λ_fe=1.0 if "
        "any --init_head_fe_ckpt / --lambda_recon_fe was set (else λ_fe=0). With --recon_path fe, λ_fe=1.0 and "
        "λ_trans=0. Auto-zero rules force a lambda to 0 with a warning if the matching head is frozen.",
    )
    g_loss.add_argument("--lambda_recon_fe",    type=float, default=None,
                        help="Weight for the post-LN FE recon loss (head_fe). None = Option B default.")
    g_loss.add_argument("--lambda_recon_trans", type=float, default=None,
                        help="Weight for the transformer-output recon loss (head_trans). None = Option B default.")
    g_loss.add_argument("--lambda_recon_proj",  type=float, default=None,
                        help="Weight for the post-extract-proj recon loss (head_proj). None = Option B default "
                             "(=1.0 if head_proj enabled, else 0.0).")
    g_loss.add_argument("--lambda_tv_fe", type=float, default=0.0,
                        help="Total-variation regularisation weight for head_fe output (0 = off). "
                             "TV = mean |x_{t+1} - x_t| over the 245-sample reconstruction.")
    g_loss.add_argument("--monitor_recon_fe",   action="store_true",
                        help="Run head_fe forward under no_grad for logging when lambda_recon_fe=0 / head_fe is frozen.")
    g_loss.add_argument("--monitor_recon_proj", action="store_true",
                        help="Run head_proj forward under no_grad for logging when lambda_recon_proj=0 / head_proj is frozen.")

    args = parser.parse_args()
    _configure_single_gpu(args.device)

    if args.mode == "train":
        run_train_mode(args)
    elif args.mode == "analyze":
        if args.ckpt_ae is None:
            print("ERROR: --ckpt_ae required for analyze mode")
            sys.exit(1)
        run_analyze_mode(args)
    else:
        run_interp_mode(args)
