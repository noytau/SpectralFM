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


def run(
    model,
    labeled_data_dir: str,
    device: str = "cpu",
    batch_size: int = 32,
    max_samples: int = 2000,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """
    Run the parameter_0 ridge probe on inputs and embeddings.

    Returns dict with label_reg_input_*, label_reg_emb_* metrics and
    label_reg_improvement_r2.
    """
    inputs, y = load_labeled_data(
        labeled_data_dir, max_samples=max_samples, seed=seed
    )
    X = _normalize_like_fairseq(inputs)
    emb = _embed(model, X, device, batch_size)

    m_in  = compute_linear_probing_metrics(X,   y, probe_type="ridge", n_folds=n_folds,
                                           task="regression", return_predictions=True)
    m_emb = compute_linear_probing_metrics(emb, y, probe_type="ridge", n_folds=n_folds,
                                           task="regression", return_predictions=True)
    y_pred_input = m_in.pop("predictions", None)
    y_pred_emb   = m_emb.pop("predictions", None)

    results = {
        **_rename_probe_metrics(m_in, "input"),
        **_rename_probe_metrics(m_emb, "emb"),
        "n_samples": len(y),
    }
    in_r2  = results.get("label_reg_input_r2")
    emb_r2 = results.get("label_reg_emb_r2")
    if in_r2 is not None and emb_r2 is not None:
        results["label_reg_improvement_r2"] = emb_r2 - in_r2
        print(f"[LabelRegression] n={len(y)}  input R²={in_r2:.4f}  "
              f"emb R²={emb_r2:.4f}  improvement={emb_r2 - in_r2:+.4f}")
    else:
        print(f"[LabelRegression] Probe failed: "
              f"{results.get('label_reg_input_error') or results.get('label_reg_emb_error')}")

    results["labels"] = y
    results["embeddings"] = emb
    results["y_pred_input"] = y_pred_input   # cross-val predictions, for true-vs-pred scatter
    results["y_pred_emb"] = y_pred_emb
    return results
