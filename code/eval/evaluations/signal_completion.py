"""
Signal Completion / Reconstruction Evaluation
----------------------------------------------
Recreates the old evaluation_runner._eval_signal_completion flow without fairseq:

  1. Ground truth: full (unmasked) forward pass → per-position embeddings
  2. Masked pass: HF Data2VecAudioModel.forward(mask_time_indices=...) replaces
     masked positions with the learned masked_spec_embed (same mechanism as
     fairseq's mask_emb — weights already remapped by checkpoint_loader)
  3. Compare GT vs masked embeddings at the masked positions:
     cosine similarity, MSE, per-bin analysis, position-error correlation
     (via metrics.compute_extended_signal_completion)

Span masking mimics fairseq's static compute_mask_indices
(mask_prob=0.65, mask_length=10 by default).

If the model has a completion_head, the legacy MSE-on-raw-signal path also runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..metrics import compute_extended_signal_completion


def make_span_mask(
    seq_len: int,
    mask_prob: float = 0.65,
    mask_length: int = 10,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Static span mask over a [seq_len] sequence, mimicking fairseq's
    compute_mask_indices with mask_selection='static'.

    Returns boolean [seq_len] array with at least one masked span.
    """
    if rng is None:
        rng = np.random.default_rng()
    mask_length = min(mask_length, seq_len)
    num_spans = max(1, int(mask_prob * seq_len / float(mask_length) + rng.random()))
    mask = np.zeros(seq_len, dtype=bool)
    starts = rng.choice(max(1, seq_len - mask_length + 1), num_spans, replace=True)
    for s in starts:
        mask[s : s + mask_length] = True
    return mask


def _completion_head_mse(model, dataloader, device) -> float | None:
    """Legacy path: predict full raw signal from masked input via completion_head."""
    if not hasattr(model, "completion_head"):
        return None
    mses = []
    with torch.no_grad():
        for batch in dataloader:
            clean = batch["data"].to(device)
            masked = batch["masked_data"].to(device)
            emb = model(input_values=masked).last_hidden_state.mean(dim=1)
            pred = model.completion_head(emb)
            mses.append(F.mse_loss(pred, clean).item())
    return float(np.mean(mses))


def run(
    df: pd.DataFrame,
    model,
    dataloader=None,
    device: str = "cpu",
    mask_prob: float = 0.65,
    mask_length: int = 10,
    max_samples: int = 200,
    seed: int = 42,
) -> dict:
    """
    Run signal reconstruction evaluation on df ('data' column, already normalised).

    Returns dict with aggregate metrics:
      completion_cos_sim, completion_mse (means over samples), per-sample
      results_df, plus averaged extended metrics (per-bin, position correlation).
    """
    model.eval()
    model.to(device)

    data = np.stack(df["data"].apply(np.array).values).astype(np.float32)
    if len(data) > max_samples:
        idx = np.random.default_rng(seed).choice(len(data), max_samples, replace=False)
        data = data[idx]
    rng = np.random.default_rng(seed)

    rows = []
    ext_metrics_acc: dict[str, list] = {}
    with torch.no_grad():
        for i in range(len(data)):
            x = torch.tensor(data[i : i + 1], dtype=torch.float32).to(device)  # [1, L]

            gt = model(input_values=x).last_hidden_state[0].cpu().numpy()      # [T, D]
            T = gt.shape[0]

            mask = make_span_mask(T, mask_prob=mask_prob, mask_length=mask_length, rng=rng)
            mask_t = torch.tensor(mask[None, :], dtype=torch.bool).to(device)  # [1, T]

            pred = model(
                input_values=x, mask_time_indices=mask_t
            ).last_hidden_state[0].cpu().numpy()                               # [T, D]

            ext = compute_extended_signal_completion(gt, pred, mask)
            if "ext_completion_error" in ext:
                continue
            for k, v in ext.items():
                ext_metrics_acc.setdefault(k, []).append(v)
            rows.append({
                "cos_sim": ext["ext_completion_cos_sim_mean"],
                "mse": ext["ext_completion_mse_mean"],
                "n_masked": int(mask.sum()),
                "seq_len": T,
            })

    if not rows:
        return {"skipped": True, "error": "No samples evaluated"}

    results_df = pd.DataFrame(rows)
    summary = {k: float(np.mean(v)) for k, v in ext_metrics_acc.items()}

    out = {
        "skipped": False,
        "completion_cos_sim": float(results_df["cos_sim"].mean()),
        "completion_mse": float(results_df["mse"].mean()),
        "n_samples": len(results_df),
        "mask_prob": mask_prob,
        "mask_length": mask_length,
        "results_df": results_df,
        "extended_summary": summary,
    }

    head_mse = _completion_head_mse(model, dataloader, device) if dataloader else None
    if head_mse is not None:
        out["completion_head_mse"] = head_mse

    print(f"[SignalCompletion] n={out['n_samples']}  "
          f"cos_sim={out['completion_cos_sim']:.4f}  mse={out['completion_mse']:.6f}")
    return out
