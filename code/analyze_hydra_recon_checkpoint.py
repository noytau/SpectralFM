#!/usr/bin/env python3
"""Quick reconstruction analysis for a fairseq Hydra checkpoint (data2vec_audio + recon_trans)."""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# fairseq + data2vec on PYTHONPATH
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (
    os.path.join(_REPO, "fairseq"),
    os.path.join(_REPO, "fairseq", "examples"),
    os.path.dirname(__file__),
):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_wavs(data_dir: str, n: int, device: torch.device):
    import glob
    import soundfile as sf

    paths = sorted(glob.glob(os.path.join(data_dir, "*.wav")))[:n]
    if not paths:
        raise FileNotFoundError(f"no wav files in {data_dir}")
    xs = []
    for p in paths:
        x, _ = sf.read(p, dtype="float32")
        xs.append(torch.from_numpy(x[:245]))
    return torch.stack(xs).to(device), paths


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="fairseq checkpoint_*.pt path")
    ap.add_argument("--data_dir", default="fairseq/data/nova_data/single_channel_100/wav")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out_dir", default="code/eval_results/hydra_recon_analysis")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from fairseq import checkpoint_utils
    from data2vec.models.data2vec_audio import Data2VecAudioModel

    device = torch.device(args.device)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.ckpt, {})
    cfg = state["cfg"]["model"]
    model = Data2VecAudioModel(cfg)
    model.load_state_dict(state["model"], strict=False)
    model.eval().to(device)

    source, paths = load_wavs(
        os.path.join(_REPO, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir,
        args.n,
        device,
    )
    target = source[:, :245].float()

    out = model(source, padding_mask=None, mask=True)
    losses = out.get("losses", {})
    print("Checkpoint losses (forward eval):", {k: float(v) for k, v in losses.items()})

    # recon_trans if enabled
    if hasattr(model, "trans_recon_decoder") and float(getattr(cfg, "lambda_recon_trans", 0)) > 0:
        enc = model(source, padding_mask=None, mask=False, features_only=True)
        x = enc["x"]
        if getattr(model.trans_recon_decoder, "needs_full_sequence", False):
            pred = model.trans_recon_decoder(x)
        else:
            tp = model._mean_pool(x, None)
            pred = model.trans_recon_decoder(tp)
        per_l2 = F.mse_loss(pred, target, reduction="none").mean(dim=1)
        print(f"recon_trans L2: mean={per_l2.mean():.4f}  min={per_l2.min():.4f}  max={per_l2.max():.4f}")

        os.makedirs(args.out_dir, exist_ok=True)
        k = min(4, source.shape[0])
        fig, axes = plt.subplots(k, 2, figsize=(10, 2.2 * k), squeeze=False)
        for i in range(k):
            axes[i, 0].plot(target[i].cpu().numpy(), color="#2c3e50", lw=1)
            axes[i, 0].set_title(os.path.basename(paths[i]), fontsize=8)
            axes[i, 0].set_ylabel("target")
            axes[i, 1].plot(pred[i].detach().cpu().numpy(), color="#e67e22", lw=1)
            axes[i, 1].set_ylabel(f"pred L2={per_l2[i]:.3f}")
        fig.suptitle(f"Hydra recon_trans — {os.path.basename(args.ckpt)}", fontsize=11)
        fig.tight_layout()
        out_png = os.path.join(args.out_dir, "recon_trans_best_worst.png")
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f"Wrote {out_png}")
    else:
        print("[!] lambda_recon_trans=0 or no trans_recon_decoder — skip recon plot")


if __name__ == "__main__":
    main()
