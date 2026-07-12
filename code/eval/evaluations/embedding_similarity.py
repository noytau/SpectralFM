"""
Embedding Similarity Evaluation
--------------------------------
For each sample, find top-k neighbors in:
  (a) raw input space  (b) embedding space
Score: fraction of embedding-space neighbors that are from the same "stack" as the query.
Compare against input-space baseline.

A high embedding score means the model groups same-stack samples closer together
than the raw signal does — i.e., the embedding captures the grouping structure.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim_sklearn


def cosine_similarity(a, b):
    return _cosine_sim_sklearn(a, b)


def extract_embeddings(model, dataloader, device: str = "cpu") -> torch.Tensor:
    """Run model on dataloader, return mean-pooled embeddings [N, D]."""
    model.eval()
    model.to(device)
    all_embs = []
    with torch.no_grad():
        for batch in dataloader:
            inp = batch.get("masked_data", batch.get("data")).to(device)
            out = model(input_values=inp)
            emb = out.last_hidden_state.mean(dim=1)  # [B, D]
            all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def run(
    df: pd.DataFrame,
    model,
    dataloader,
    k: int = 5,
    device: str = "cpu",
) -> dict:
    """
    Run embedding similarity evaluation.

    Returns dict with:
      - results_df: per-sample DataFrame with neighbor info
      - embedding_stack_match_rate: avg fraction of embedding neighbors from same stack
      - input_stack_match_rate: avg fraction of input neighbors from same stack
      - match_score_avg: avg (emb_matches - input_matches) normalized to [0,100]
    """
    embeddings = extract_embeddings(model, dataloader, device=device)

    data_col = "data"
    input_matrix = np.stack(df[data_col].apply(np.array).values)
    embedding_matrix = embeddings.numpy()

    rows = []
    for idx in range(len(df)):
        query_stack = df.iloc[idx]["stack_idx"]
        query_input = input_matrix[idx : idx + 1]
        query_emb = embedding_matrix[idx : idx + 1]

        input_sims = cosine_similarity(query_input, input_matrix)[0]
        emb_sims = cosine_similarity(query_emb, embedding_matrix)[0]

        # Exclude self
        input_sims[idx] = -np.inf
        emb_sims[idx] = -np.inf

        topk_input = input_sims.argsort()[::-1][:k]
        topk_emb = emb_sims.argsort()[::-1][:k]

        input_stack_matches = [i for i in topk_input if df.iloc[i]["stack_idx"] == query_stack]
        emb_stack_matches = [i for i in topk_emb if df.iloc[i]["stack_idx"] == query_stack]

        # Same-stack neighbors (for reference)
        same_stack = df[df["stack_idx"] == query_stack].drop(index=idx, errors="ignore")
        same_stack = same_stack.copy()
        same_stack["_sim"] = [input_sims[i] for i in same_stack.index]
        same_stack_top = same_stack.nlargest(k, "_sim")

        rows.append({
            "index": idx,
            "stack_idx": query_stack,
            "embedding_neighbors": topk_emb.tolist(),
            "embedding_similarities": emb_sims[topk_emb].tolist(),
            "embedding_stack_matches": emb_stack_matches,
            "input_neighbors": topk_input.tolist(),
            "input_similarities": input_sims[topk_input].tolist(),
            "input_stack_matches": input_stack_matches,
            "same_stack_neighbors": same_stack_top.index.tolist(),
            "same_stack_similarities": same_stack_top["_sim"].tolist(),
        })

    results_df = pd.DataFrame(rows)

    emb_rate = results_df["embedding_stack_matches"].apply(len).mean() / k
    inp_rate = results_df["input_stack_matches"].apply(len).mean() / k

    results_df["match_diff"] = results_df.apply(
        lambda r: len(set(r["embedding_stack_matches"])) - len(set(r["input_stack_matches"])),
        axis=1,
    )
    results_df["match_score"] = ((results_df["match_diff"] + k) / (2 * k)) * 100

    sim_dists = _compute_similarity_distributions(df, embedding_matrix, input_matrix)

    return {
        "results_df": results_df,
        "embedding_stack_match_rate": emb_rate,
        "input_stack_match_rate": inp_rate,
        "match_score_avg": results_df["match_score"].mean(),
        "embeddings": embeddings,
        "inputs": input_matrix,   # raw signals, for neighbor example plots
        "similarity_distributions": sim_dists,
    }


def _compute_similarity_distributions(
    df: pd.DataFrame,
    embedding_matrix: np.ndarray,
    input_matrix: np.ndarray,
    n_random: int = 500,
) -> dict:
    """
    Compute cosine similarity distributions for three pair types:
      intra_stack    — both samples from the SAME stack
      random_inter   — randomly selected samples from DIFFERENT stacks
      preset_inter   — same position-within-stack, from DIFFERENT stacks
    Returns dict with {pair_type}_{space} arrays, space in {emb, inp}.
    """
    n = len(df)
    df = df.reset_index(drop=True)

    # Full pairwise similarity matrices (faster than row-by-row)
    emb_sim_full = _cosine_sim_sklearn(embedding_matrix)  # [N, N]
    inp_sim_full = _cosine_sim_sklearn(input_matrix)       # [N, N]

    stack_arr = df["stack_idx"].to_numpy()
    df_cp = df.copy()
    df_cp["_pos"] = df_cp.groupby("stack_idx").cumcount()
    pos_arr = df_cp["_pos"].to_numpy()

    # --- Intra-stack: upper-triangle pairs where stack[i] == stack[j] ---
    ii, jj = np.triu_indices(n, k=1)
    same_mask = stack_arr[ii] == stack_arr[jj]
    diff_mask = ~same_mask

    intra_emb = emb_sim_full[ii[same_mask], jj[same_mask]]
    intra_inp = inp_sim_full[ii[same_mask], jj[same_mask]]

    # --- Random inter-stack: uniform sample from diff-stack pairs ---
    diff_ii, diff_jj = ii[diff_mask], jj[diff_mask]
    rng = np.random.default_rng(42)
    n_sample = min(n_random, len(diff_ii))
    sel = rng.choice(len(diff_ii), n_sample, replace=False)
    random_inter_emb = emb_sim_full[diff_ii[sel], diff_jj[sel]]
    random_inter_inp = inp_sim_full[diff_ii[sel], diff_jj[sel]]

    # --- Preset-index inter-stack: same position across different stacks ---
    preset_inter_emb, preset_inter_inp = [], []
    for pos in np.unique(pos_arr):
        pos_idx = np.where(pos_arr == pos)[0]
        pi, pj = np.triu_indices(len(pos_idx), k=1)
        gi, gj = pos_idx[pi], pos_idx[pj]
        diff = stack_arr[gi] != stack_arr[gj]
        preset_inter_emb.append(emb_sim_full[gi[diff], gj[diff]])
        preset_inter_inp.append(inp_sim_full[gi[diff], gj[diff]])

    return {
        "sim_matrix_emb":     emb_sim_full,   # [N, N] full pairwise cosine sim (embedding)
        "sim_matrix_inp":     inp_sim_full,   # [N, N] full pairwise cosine sim (input)
        "intra_stack_emb":    intra_emb,
        "intra_stack_inp":    intra_inp,
        "random_inter_emb":   random_inter_emb,
        "random_inter_inp":   random_inter_inp,
        "preset_inter_emb":   np.concatenate(preset_inter_emb) if preset_inter_emb else np.array([]),
        "preset_inter_inp":   np.concatenate(preset_inter_inp) if preset_inter_inp else np.array([]),
    }
