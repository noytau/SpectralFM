"""
Noise Robustness Evaluation
-----------------------------
For each sample, add several types of noise to the raw input.
Run the model on both clean and noisy inputs.
Score: cosine similarity between clean embedding and noisy embedding.
Higher = model is more robust to that noise type.

Noise types: gaussian_std, gaussian_mean, shot_low, shot_high, gain_low, gain_high.
"""
from __future__ import annotations
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_sim


NOISE_CONFIGS = {
    "gaussian_std":  lambda x: x + np.random.normal(0, 0.01, size=x.shape),
    "gaussian_mean": lambda x: x + np.random.normal(2, 0.001, size=x.shape),
    "shot_low":      lambda x: np.random.poisson((x - x.min() + 1e-4) * 0.1) / 0.1 - 1e-4,
    "shot_high":     lambda x: np.random.poisson((x - x.min() + 1e-4) * 0.05) / 0.05 - 1e-4,
    "gain_low":      lambda x: x * np.random.normal(1, 0.05),
    "gain_high":     lambda x: x * np.random.normal(1, 0.1),
}


def _embed_batch(model, data: np.ndarray, device: str, batch_size: int = 16) -> np.ndarray:
    """Run model on numpy array [N, L], return embeddings [N, D]."""
    model.eval()
    model.to(device)
    all_embs = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            chunk = torch.tensor(data[start : start + batch_size], dtype=torch.float32)
            chunk = chunk.to(device)  # [B, L]
            out = model(input_values=chunk)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()  # [B, D]
            all_embs.append(emb)
    return np.concatenate(all_embs, axis=0)


def run(
    df: pd.DataFrame,
    model,
    device: str = "cpu",
    noise_types: list | None = None,
    batch_size: int = 16,
) -> dict:
    """
    Run noise robustness evaluation.
    Returns dict with:
      - results_df: per-sample per-noise cosine similarity
      - summary: {noise_type: mean_cosine_similarity}
    """
    if noise_types is None:
        noise_types = list(NOISE_CONFIGS.keys())

    clean_data = np.stack(df["data"].apply(np.array).values)  # [N, L]
    clean_embs = _embed_batch(model, clean_data, device, batch_size)

    summary = {}
    per_sample = {nt: [] for nt in noise_types}
    noisy_data_by_type = {}

    for noise_type in noise_types:
        fn = NOISE_CONFIGS[noise_type]
        noisy_data = np.stack([fn(row.astype(float)) for row in clean_data])
        noisy_data_by_type[noise_type] = noisy_data
        noisy_embs = _embed_batch(model, noisy_data, device, batch_size)

        # Cosine similarity per sample (clean_emb vs noisy_emb)
        sims = [
            sk_cosine_sim(clean_embs[i : i + 1], noisy_embs[i : i + 1])[0, 0]
            for i in range(len(clean_embs))
        ]
        per_sample[noise_type] = sims
        summary[noise_type] = float(np.mean(sims))
        print(f"[NoiseRobustness] {noise_type:20s}: mean cosine sim = {summary[noise_type]:.4f}")

    results_df = pd.DataFrame(per_sample)
    results_df.insert(0, "stack_idx", df["stack_idx"].values)
    results_df.insert(0, "index", np.arange(len(results_df)))

    return {
        "results_df": results_df,
        "summary": summary,
        "clean_data": clean_data,           # [N, L] for best/worst example plots
        "noisy_data": noisy_data_by_type,   # {noise_type: [N, L]}
    }
