"""
Label Regression Evaluation (parameter_0 linear probe)
-------------------------------------------------------
Ridge-CV probe predicting parameter_0 from raw inputs vs transformer embeddings.
Ported from Geoffrey eval_label_regression.py.

Metrics are prefixed label_reg_input_* and label_reg_emb_*; improvement is
label_reg_improvement_r2 = emb R² − input R².
"""
from __future__ import annotations

import numpy as np
import torch

from ..data_loader import load_labeled_data
from ..metrics import compute_linear_probing_metrics


def _rename_probe_metrics(m: dict, kind: str) -> dict:
    out = {}
    for k, v in m.items():
        if isinstance(v, str):
            out[f"label_reg_{kind}_error"] = v
            continue
        if not isinstance(v, (int, float, np.floating)):
            continue
        tail = k.replace("downstream_ridge_probe_", "")
        out[f"label_reg_{kind}_{tail}"] = float(v)
    return out


def _normalize_like_fairseq(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean(axis=1, keepdims=True)
    std  = arr.std(axis=1,  keepdims=True) + 1e-8
    return ((arr - mean) / std).astype(np.float32)


def _embed(model, data: np.ndarray, device: str, batch_size: int = 32) -> np.ndarray:
    model.eval()
    model.to(device)
    embs = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.tensor(data[i : i + batch_size], dtype=torch.float32).to(device)
            out = model(input_values=batch)
            embs.append(out.last_hidden_state.mean(dim=1).cpu().numpy())
    return np.concatenate(embs, axis=0)


# Multi-channel configurations, matching label_reg_evaluation.py comp_rows:
# 1-comp: raw 245 / emb 768;  2-comp: raw 490 / emb 1536;  3-comp: raw 735 / emb 2304.
COMP_CONFIGS = [((0,), ""), ((0, 1), "_2c"), ((0, 1, 2), "_3c")]


def run(
    model,
    labeled_data_dir: str,
    device: str = "cpu",
    batch_size: int = 32,
    max_samples: int = 2000,
    n_folds: int = 5,
    seed: int = 42,
    comp_configs: list = None,
) -> dict:
    """
    Run the parameter_0 ridge probe on inputs and embeddings, for each
    multi-channel configuration (components of the same spectrum concatenated —
    raw signals side by side, per-component embeddings concatenated).

    Data is loaded once with all 3 components (only spectra having all three are
    kept) so every configuration is evaluated on the SAME spectra.

    Returns dict with label_reg_input_* / label_reg_emb_* metrics per config
    (suffix '' = 1-comp, '_2c' = comps 0+1, '_3c' = comps 0+1+2) and
    label_reg_improvement_r2 variants.
    """
    if comp_configs is None:
        comp_configs = COMP_CONFIGS

    max_comps = tuple(sorted({c for comps, _ in comp_configs for c in comps}))
    inputs, y = load_labeled_data(
        labeled_data_dir, max_samples=max_samples, seed=seed, comps=max_comps
    )                                                    # [N, n_comps, L]

    # Normalize + embed each component once; slice per config
    n, n_comps, L = inputs.shape
    X_flat = _normalize_like_fairseq(inputs.reshape(n * n_comps, L))
    emb_flat = _embed(model, X_flat, device, batch_size)  # [N*n_comps, D]
    X_by_comp   = X_flat.reshape(n, n_comps, L)
    emb_by_comp = emb_flat.reshape(n, n_comps, -1)
    comp_pos = {c: i for i, c in enumerate(max_comps)}

    results = {"n_samples": len(y), "labels": y}
    for comps, suffix in comp_configs:
        cols = [comp_pos[c] for c in comps]
        X   = X_by_comp[:, cols, :].reshape(n, -1)        # [N, L*k]
        emb = emb_by_comp[:, cols, :].reshape(n, -1)      # [N, D*k]

        m_in  = compute_linear_probing_metrics(X,   y, probe_type="ridge", n_folds=n_folds,
                                               task="regression", return_predictions=True)
        m_emb = compute_linear_probing_metrics(emb, y, probe_type="ridge", n_folds=n_folds,
                                               task="regression", return_predictions=True)
        y_pred_input = m_in.pop("predictions", None)
        y_pred_emb   = m_emb.pop("predictions", None)

        for k, v in _rename_probe_metrics(m_in, "input").items():
            results[k + suffix] = v
        for k, v in _rename_probe_metrics(m_emb, "emb").items():
            results[k + suffix] = v

        in_r2  = results.get(f"label_reg_input_r2{suffix}")
        emb_r2 = results.get(f"label_reg_emb_r2{suffix}")
        if in_r2 is not None and emb_r2 is not None:
            results[f"label_reg_improvement_r2{suffix}"] = emb_r2 - in_r2
            print(f"[LabelRegression] comps={comps}  n={len(y)}  input R²={in_r2:.4f}  "
                  f"emb R²={emb_r2:.4f}  improvement={emb_r2 - in_r2:+.4f}")

        if suffix == "":  # 1-comp predictions feed the true-vs-pred scatter
            results["y_pred_input"] = y_pred_input
            results["y_pred_emb"] = y_pred_emb
            results["embeddings"] = emb

    return results
