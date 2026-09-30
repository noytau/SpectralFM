"""
Four-stage representation extraction in ONE forward pass.

Taps: input (the signal itself), fe (pre-LayerNorm conv output), proj
(post-projection, what the transformer consumes), transformer (encoder
output). A fifth diagnostic column, extract_features (post-LayerNorm FE,
what structured_similarity.py actually taps), is captured too since it
differs from the pre-LN `fe` column.

Costs ONE model(input_values=...) call per batch via forward hooks on
model.feature_extractor and model.feature_projection.projection — the
`_extract_all_representations` reference in structured_similarity.py costs
two FE passes per batch (one manual, one inside model(...)); this avoids
that duplication.

See plan.md's "Reuse" table and step 1 for the design this implements.
"""
from __future__ import annotations

import numpy as np
import torch

STAGES = ("input", "fe", "extract_features", "proj", "transformer")

POOLINGS = ("mean", "mean_std", "mean_max_min", "segment4", "first_last")


def _pool(seq: torch.Tensor, pooling: str) -> torch.Tensor:
    """
    seq: [B, T, C] (time-major). Returns [B, C * k] for some small k
    depending on the pooling scheme.
    """
    if pooling == "mean":
        return seq.mean(dim=1)
    if pooling == "mean_std":
        return torch.cat([seq.mean(dim=1), seq.std(dim=1)], dim=-1)
    if pooling == "mean_max_min":
        return torch.cat([seq.mean(dim=1), seq.amax(dim=1), seq.amin(dim=1)], dim=-1)
    if pooling == "segment4":
        T = seq.shape[1]
        edges = np.linspace(0, T, 5).astype(int)
        segs = [seq[:, edges[i]:max(edges[i] + 1, edges[i + 1]), :].mean(dim=1)
                for i in range(4)]
        return torch.cat(segs, dim=-1)
    if pooling == "first_last":
        return torch.cat([seq[:, 0, :], seq[:, -1, :]], dim=-1)
    raise ValueError(f"unknown pooling {pooling!r}")


def _pool_channels_first(seq: torch.Tensor, pooling: str) -> torch.Tensor:
    """seq: [B, C, T] (channel-major, the raw FE tap). Pool over the last dim."""
    return _pool(seq.transpose(1, 2), pooling)


@torch.no_grad()
def extract_stage_reps(
    model,
    x: np.ndarray,
    device: str = "cpu",
    batch_size: int = 32,
    pooling: str = "mean",
) -> dict:
    """
    x: [N, L] normalized 1D signals (one component's worth; caller loops
       over components and concatenates).

    Returns a dict of stage -> [N, D_stage] numpy arrays, D_stage depending
    on the pooling scheme (D=C for 'mean', 2C for 'mean_std'/'first_last',
    3C for 'mean_max_min', 4C for 'segment4').

    Tensor shapes for a 245-sample input (verified empirically, see plan.md):
      input        [245]        -> pooled with itself (identity; 'mean' keeps [1])
      fe           [512, 47]    channel-major, PRE-LayerNorm
      extract_features [47,512] time-major, POST-LayerNorm (what
                                  structured_similarity.py taps)
      proj         [47, 768]    post feature_projection.projection
      transformer  [47, 768]    out.last_hidden_state
    """
    model.eval()
    model.to(device)
    t = torch.from_numpy(x).float()

    out = {s: [] for s in STAGES}
    captured = {}
    handles = [
        model.feature_projection.projection.register_forward_hook(
            lambda _m, _i, o: captured.__setitem__("proj_seq", o)),
    ]
    try:
        for i in range(0, len(t), batch_size):
            batch = t[i:i + batch_size].to(device)   # [B, 245]
            if batch.dim() == 1:
                batch = batch.unsqueeze(0)

            # input stage: no pooling needed beyond identity, but keep the
            # same [B, D] contract as everything else.
            out["input"].append(batch.cpu().numpy())

            fe_out = model.feature_extractor(batch)              # [B, 512, 47], pre-LN
            out["fe"].append(_pool_channels_first(fe_out, pooling).cpu().numpy())

            fe_t = fe_out.transpose(1, 2)                        # [B, 47, 512]
            ef = model.feature_projection.layer_norm(fe_t)       # post-LN
            out["extract_features"].append(_pool(ef, pooling).cpu().numpy())

            hf_out = model(input_values=batch)                   # triggers the proj hook
            proj_seq = captured.pop("proj_seq")                  # [B, 47, 768]
            out["proj"].append(_pool(proj_seq, pooling).cpu().numpy())

            out["transformer"].append(
                _pool(hf_out.last_hidden_state, pooling).cpu().numpy())
    finally:
        for h in handles:
            h.remove()

    return {s: np.concatenate(v, axis=0) for s, v in out.items()}


def stage_shapes(model, device: str = "cpu") -> dict:
    """One dummy [1, 245] forward to report tap shapes; used by the
    verification gate (plan.md gate 4)."""
    x = np.zeros((1, 245), dtype=np.float32)
    reps = extract_stage_reps(model, x, device=device, batch_size=1, pooling="mean")
    return {s: v.shape for s, v in reps.items()}
