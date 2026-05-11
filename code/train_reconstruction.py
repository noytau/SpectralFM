"""
Reconstruction training and analysis for SpectralFM.

  - train --recon_path fe:          FE + MirrorDecoder (ConvTranspose1d), optional FE init from --ckpt
  - train --recon_path transformer: Frozen ``Data2VecAudioModel`` (SpectralFM cfg) + weights merged from
                                      ``--ckpt`` like Fairseq ``build_model`` (not ``load_model_ensemble``), so
                                      ``base_libri_official.pt``-style tensors (e.g. ``feature_extractor.*``) load
                                      into the audio skeleton; remaining layers stay init unless keys match.
  - analyze:                        Load AE ckpt; for transformer also pass --ckpt (fairseq backbone)
  - interp:                         Pretrained interp decoder (full fairseq model)

Usage:
    python code/train_reconstruction.py --mode train --recon_path fe --lr 1e-4 --n_samples 1000 --steps 10000
    python code/train_reconstruction.py --mode train --recon_path transformer --ckpt none --n_samples 1000 --steps 2000
    # Default: --batch_size 512 --grad_accum_steps 4 (micro-batch 128 per forward; effective 512 per optimizer step)
    python code/train_reconstruction.py --mode train --recon_path transformer --ckpt /path/to/checkpoint_best.pt ...
    python code/train_reconstruction.py --mode analyze --ckpt_ae path/to/ckpt_tr_....pt --ckpt path/to/checkpoint_best.pt \\
        --recon_path transformer --datasets single_channel_10k
"""
import sys, os, warnings, logging, math, torch, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from importlib import util as _ilu

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
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_state = state.get("model", state)
        if not isinstance(model_state, dict):
            model_state = {}
        fe_state = {k.replace("feature_extractor.", "", 1): v
                    for k, v in model_state.items() if k.startswith("feature_extractor.")}
        if not fe_state:
            _pref = "modality_encoders.AUDIO.local_encoder."
            fe_state = {k[len(_pref):]: v for k, v in model_state.items()
                        if k.startswith(_pref)}
        tgt_enc = encoder.state_dict()
        fe_ok = {k: v for k, v in fe_state.items()
                 if k in tgt_enc and v.shape == tgt_enc[k].shape}
        if len(fe_ok) < len(fe_state):
            print(f"[!] FE init: skipped {len(fe_state) - len(fe_ok)} tensors (shape mismatch vs SpectralFM FE)")
        if fe_ok:
            encoder.load_state_dict(fe_ok, strict=False)
            print(f"[+] Loaded FE weights from {ckpt_path} ({len(fe_ok)} tensors applied)")
        else:
            print(f"[!] No compatible FE tensors in {ckpt_path} — using random init")
        ln_state = {k.replace("layer_norm.", "", 1): v
                    for k, v in model_state.items() if k.startswith("layer_norm.")}
        tgt_ln = layer_norm.state_dict()
        ln_ok = {k: v for k, v in ln_state.items()
                 if k in tgt_ln and v.shape == tgt_ln[k].shape}
        if ln_ok:
            layer_norm.load_state_dict(ln_ok, strict=False)
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
    """Map transformer encoder output [B, C, T] → spectrogram [B, 1, 245] without mean-pooling.

    Projects channels C→512 with a 1×1 conv, then reuses ``MirrorDecoder`` (same temporal upsampling
    as the FE autoencoder). T must match the subsampled FE length (47 for the default 245-bin conv stack).
    """

    def __init__(self, encoder_embed_dim: int = 768, mid_channels: int = 512):
        super().__init__()
        self.encoder_embed_dim = int(encoder_embed_dim)
        self.stem = nn.Conv1d(self.encoder_embed_dim, mid_channels, kernel_size=1, bias=True)
        self.decoder = MirrorDecoder()

    def forward(self, x_bt_c):
        """x_bt_c: [B, T, C] from fairseq ``features_only`` encoder output."""
        x = x_bt_c.transpose(1, 2).contiguous()
        x = self.stem(x)
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


def build_random_data2vec_audio(device):
    """Randomly initialized ``Data2VecAudioModel`` (SpectralFM 245-bin conv stack, no base checkpoint)."""
    return build_data2vec_audio_backbone(device, None)


def _freeze_module_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


@torch.no_grad()
def _transformer_latent_btc(model, source_245: torch.Tensor):
    """Encoder stack with ``mask=False``, ``features_only=True`` → [B, T, C]."""
    model.eval()
    out = model(
        source_245,
        padding_mask=None,
        mask=False,
        features_only=True,
    )
    return out["x"]


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

    def __init__(self, manifest_path, max_samples=None):
        with open(manifest_path) as f:
            self.root = f.readline().strip()
            lines = f.readlines()
        if max_samples:
            lines = lines[:max_samples]
        self.fnames = []
        for ln in lines:
            parts = ln.strip().split("\t")
            self.fnames.append(parts[0])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        import soundfile as sf
        wav, _ = sf.read(os.path.join(self.root, self.fnames[idx]), dtype="float32")
        return torch.from_numpy(wav).float()


def exp_tag(args):
    rp = getattr(args, "recon_path", "fe")
    prefix = "tr_" if rp == "transformer" else ""
    base = f"{prefix}lr{args.lr}_n{args.n_samples}_s{args.steps}"
    if args.warmup > 0:
        base += f"_w{args.warmup}"
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

def _run_train_transformer_mirror(args):
    """Train 1×1 stem + MirrorDecoder on frozen (or random) data2vec_audio encoder outputs."""
    tag = exp_tag(args)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_arg = (args.ckpt or "").strip()
    if not ckpt_arg or ckpt_arg.lower() == "none":
        backbone = build_data2vec_audio_backbone(args.device, None)
        ckpt_label = "random_backbone"
    else:
        if not os.path.isfile(ckpt_arg):
            raise FileNotFoundError(f"--ckpt not found: {ckpt_arg}")
        backbone = build_data2vec_audio_backbone(args.device, ckpt_arg)
        ckpt_label = ckpt_arg

    _freeze_module_params(backbone)

    enc_dim = int(backbone.cfg.encoder_embed_dim)
    decoder = TransformerMirrorDecoder(encoder_embed_dim=enc_dim).to(args.device)
    for p in decoder.parameters():
        p.requires_grad = True

    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n{'='*60}")
    print(f"TRAIN transformer+mirror  [{tag}]")
    print(f"Decoder (stem+mirror) params: {dec_params:,}")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
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
            f"(dataset or n too small for 512 with accum {args.grad_accum_steps})\n"
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
                "dec_params": dec_params,
                "recon_path": "transformer",
                "backbone_ckpt": ckpt_label,
                "encoder_embed_dim": enc_dim,
                "manifest": getattr(args, "manifest", None),
                "data_dir": getattr(args, "data_dir", None),
                "exp_tag": tag,
            },
        )

    decoder.train()
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
                source_batch = _next_lazy_batch()
            else:
                idx = torch.randint(0, n, (micro_bs,), device=args.device)
                source_batch = source_all[idx]
            target = source_batch[:, :245].float()
            with torch.no_grad():
                lat = _transformer_latent_btc(backbone, source_batch)
            lat = lat.detach()
            recon, _ = decoder(lat)
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
                gn = sum(p.grad.norm().item() ** 2 for p in decoder.parameters()
                         if p.grad is not None) ** 0.5
                log_d["train/grad_norm"] = gn
            wb_run.log(log_d, step=step)

        if step % log_interval == 0 or step == 1:
            print(f"  step {step:5d}/{args.steps}  L2 = {loss_sum:.6f}  lr = {lr_now:.2e}  epoch = {epoch:.1f}")

    ckpt_path = str(Path(args.out_dir) / f"ckpt_{tag}.pt")
    torch.save({
        "recon_path": "transformer",
        "encoder_embed_dim": enc_dim,
        "transformer_mirror": decoder.state_dict(),
        "losses": losses,
        "lrs": lrs,
        "tag": tag,
        "lr": args.lr,
        "warmup": args.warmup,
        "n_samples": args.n_samples,
        "steps": args.steps,
        "backbone_ckpt": ckpt_label,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "micro_batch_size": micro_bs,
        "effective_batch_size": eff_bs,
    }, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")
    print(f"Final train L2: {losses[-1]:.6f}")
    if wb_run:
        wb_run.log({"train/final_loss": losses[-1]})
        wb_run.finish()


def run_train_mode(args):
    if getattr(args, "recon_path", "fe") == "transformer":
        return _run_train_transformer_mirror(args)

    tag = exp_tag(args)
    os.makedirs(args.out_dir, exist_ok=True)

    encoder, layer_norm = build_fe_standalone(args.device, args.ckpt)
    decoder = MirrorDecoder().to(args.device)

    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n{'='*60}")
    print(f"TRAIN  [{tag}]")
    print(f"Encoder: {enc_params:,}  Decoder: {dec_params:,}  Total: {enc_params+dec_params:,}")
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
                "enc_params": enc_params, "dec_params": dec_params,
                "ckpt_init": args.ckpt or "random",
                "loader": "lazy" if use_lazy else "preload",
                "recon_path": "fe",
                "manifest": getattr(args, "manifest", None),
                "data_dir": getattr(args, "data_dir", None),
                "exp_tag": tag,
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
    """Load ``TransformerMirrorDecoder`` weights from a train-mode transformer checkpoint."""
    ckpt = torch.load(ckpt_ae_path, map_location=device)
    enc_dim = int(ckpt.get("encoder_embed_dim", 768))
    decoder = TransformerMirrorDecoder(encoder_embed_dim=enc_dim).to(device)
    decoder.load_state_dict(ckpt["transformer_mirror"])
    decoder.eval()
    return decoder, ckpt


def _eval_transformer_mirror_on_samples(backbone, decoder, source, device):
    """Backbone (fairseq) + stem+mirror decoder; same return shape tuple as ``_eval_on_samples``."""
    target = source[:, :245].float()
    with torch.no_grad():
        lat = _transformer_latent_btc(backbone, source)
        recon, intermediates = decoder(lat)
        pred = recon.squeeze(1)
        per_sample_l2 = ((pred - target) ** 2).mean(dim=1)
    fe_dummy = torch.zeros(source.shape[0], 512, 47, device=source.device, dtype=source.dtype)
    fe_norm_dummy = fe_dummy.transpose(1, 2)
    return per_sample_l2, pred, target, fe_dummy, fe_norm_dummy, intermediates


def _run_analyze_transformer_mirror(args):
    """Analyze datasets using frozen fairseq backbone + trained ``TransformerMirrorDecoder``."""
    ck = (args.ckpt or "").strip()
    if not ck or ck.lower() == "none" or not os.path.isfile(ck):
        print("ERROR: --recon_path transformer analyze requires existing --ckpt (fairseq .pt tensor dict)")
        sys.exit(1)

    decoder, ckpt = _load_transformer_mirror_from_ckpt(args.ckpt_ae, args.device)
    backbone = build_data2vec_audio_backbone(args.device, ck)
    _freeze_module_params(backbone)

    tag = ckpt["tag"]

    print(f"\n{'='*60}")
    print(f"ANALYZE transformer+mirror  [{tag}]  decoder={args.ckpt_ae}  backbone={ck}")
    print(f"{'='*60}")

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
            backbone, decoder, source_stat, args.device)
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
    print(f"ANALYZE  [{tag}]  from {args.ckpt_ae}")
    print(f"{'='*60}")

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
        help="fe: CNN+MirrorDecoder; transformer: frozen data2vec_audio latent + stem+MirrorDecoder",
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
    parser.add_argument("--k", type=int, default=5, help="Number of samples to plot in analyze mode")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated dataset names for analyze mode "
                             "(e.g. single_channel_10k,multi_channel,labeled_data)")
    parser.add_argument("--n_stat", type=int, default=100,
                        help="Number of samples for statistical evaluation per dataset")
    parser.add_argument("--out_dir", default="autoencoder_experiments")
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
    args = parser.parse_args()
    _configure_single_gpu(args.device)

    if args.mode == "train":
        run_train_mode(args)
    elif args.mode == "analyze":
        if args.ckpt_ae is None:
            print("ERROR: --ckpt_ae required for analyze mode")
            sys.exit(1)
        if getattr(args, "recon_path", "fe") == "transformer":
            ck = (args.ckpt or "").strip()
            if not ck or ck.lower() == "none" or not os.path.isfile(ck):
                print("ERROR: transformer analyze needs a real fairseq checkpoint path via --ckpt")
                sys.exit(1)
        run_analyze_mode(args)
    else:
        run_interp_mode(args)
