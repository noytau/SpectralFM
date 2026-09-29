"""
Collect every number the backbone-comparison report needs into one dict.

Reads, for each backbone in BACKBONES and each label set:
  <eval_dir>/label_probe_regression_<tag>/label_probe/<set>/      (one per set)
  <eval_dir>/label_probe_regression_merged_<tag>/label_probe/labeled_regression_all/
  <eval_dir>/label_probe_regression_labeled_data_<tag>/label_probe/labeled_data/
each holding nested_results.json + nested_oof.npz (nested.py),
label_probe_results.json (canary) and recipe_panel.json (label efficiency).

Headline numbers are the nested-CV scores. Backbone-vs-backbone and
embedding-vs-raw differences are paired: every arm on a label set shares the
same outer folds, so the difference's SD comes from resampling the same
spectra for both (nested.paired_delta).
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os

import numpy as np

from .nested import paired_delta
from .readouts import stage_display_name

# The checkpoints being compared. `tag` names the run directories; the rest
# is shown in the report's backbone table. mix: pretraining-data shares (%).
BACKBONES = [
    dict(tag="ref_feb25", short="Feb-25 ref", name="Feb-25 reference",
         ckpt="runai/runai_long_train_2026-02-25_13-46-46.pt",
         stage="data2vec only (no recon heads)", data="single_channel_all",
         mix=[("single-channel", 100.0)], n_pretrain=9_099_930,
         projector="linear 512→768", updates="13.5k"),
    dict(tag="jointScratch_singlech", short="jointScratch", name="jointScratch_singlech",
         ckpt="transPretrain_scratch/jointScratch_all30k.pt",
         stage="joint from random init (d2v + 3 recon heads)", data="single_channel_all",
         mix=[("single-channel", 100.0)], n_pretrain=9_099_930,
         projector="MLP 512→2048→768 + LN", updates="30k"),
    dict(tag="s4_singlech", short="s4 singlech", name="step4 pretrain_singlech",
         ckpt="final_checkpoints/step4/pretrain_singlech.pt",
         stage="step 4 joint fine-tune", data="single_channel_all",
         mix=[("single-channel", 100.0)], n_pretrain=9_099_930,
         projector="MLP 512→2048→768 + LN", updates="30k"),
    dict(tag="s4_sampled", short="s4 sampled", name="step4 pretrain_single_multi_sampled",
         ckpt="final_checkpoints/step4/pretrain_single_multi_sampled.pt",
         stage="step 4 joint fine-tune", data="single + multipos + sampled 0.15%",
         mix=[("single-channel", 79.97), ("multi-channel (pos)", 19.87), ("sampled", 0.15)],
         n_pretrain=11_378_718, projector="MLP 512→2048→768 + LN", updates="30k"),
    dict(tag="s4_sampledUp", short="s4 sampledUp", name="step4 pretrain_single_multi_sampledUp",
         ckpt="final_checkpoints/step4/pretrain_single_multi_sampledUp.pt",
         stage="step 4 joint fine-tune", data="single + multipos + sampled ×34",
         mix=[("single-channel", 76.11), ("multi-channel (pos)", 18.91), ("sampled", 4.972)],
         n_pretrain=11_955_723, projector="MLP 512→2048→768 + LN", updates="30k"),
]
MERGED = "all merged"
LD = "labeled_data"


def _run_dir(eval_dir, tag, ds):
    if ds == MERGED:
        return os.path.join(eval_dir, f"label_probe_regression_merged_{tag}", "label_probe", "labeled_regression_all")
    if ds == LD:
        return os.path.join(eval_dir, f"label_probe_regression_labeled_data_{tag}", "label_probe", "labeled_data")
    return os.path.join(eval_dir, f"label_probe_regression_{tag}", "label_probe", ds)


def _choice(chosen, with_arm=True):
    """Most frequent (block, recipe) over the outer folds, with its count."""
    c = collections.Counter((x["arm"], x["norm"], x["probe"]) if with_arm else (x["norm"], x["probe"])
                            for x in chosen)
    top, k = c.most_common(1)[0]
    blocks = collections.Counter(x["arm"] for x in chosen)
    return {"top": list(top), "count": k, "of": len(chosen),
            "block_counts": dict(blocks.most_common())}


def _crossing(ns, gaps):
    """Where (if anywhere) the embedding's median overtakes raw input's,
    log-interpolated between measured budgets."""
    if all(g >= 0 for g in gaps):
        return "embedding ahead at every budget"
    if all(g < 0 for g in gaps):
        return "raw ahead at every budget"
    last_neg = max(i for i, g in enumerate(gaps) if g < 0)
    if last_neg == len(gaps) - 1:
        return "mixed; raw ahead at the full pool"
    i = last_neg
    f = -gaps[i] / (gaps[i + 1] - gaps[i])
    x = 10 ** (math.log10(ns[i]) + f * (math.log10(ns[i + 1]) - math.log10(ns[i])))
    return f"embedding ahead from n≈{x:,.0f}"


def _panel(path):
    if not os.path.isfile(path):
        return None
    p = json.load(open(path))["by_n_comp"]["1"]
    rungs = sorted(p["selected"], key=int)
    raw = {k: {"med": p["selected"][k]["r2_median"], "recipe": p["selected"][k]["recipe"]} for k in rungs}
    lay = {}
    for k in rungs:
        name = max(p["layer_curves"], key=lambda n: p["layer_curves"][n][k]["r2_median"])
        v = p["layer_curves"][name][k]
        lay[k] = {"med": v["r2_median"], "layer": name, "recipe": v.get("recipe")}
    # Hindsight bound: each arm's best recipe picked on the scoring split
    # itself -- optimistic for both arms equally.
    raw_h = {k: max(v["report"][k]["r2_median"] for v in p["panel"].values()) for k in rungs}
    lay_h = {k: max(max(rp[k] for rp in lp.values()) for lp in p["layer_panels"].values()) for k in rungs}
    ns = [int(k) for k in rungs]
    return {"rungs": ns, "raw": raw, "layer": lay, "raw_hind": raw_h, "layer_hind": lay_h,
            "crossing": _crossing(ns, [lay[k]["med"] - raw[k]["med"] for k in rungs]),
            "crossing_hind": _crossing(ns, [lay_h[k] - raw_h[k] for k in rungs]),
            "n_eval_select": p.get("n_eval_select"), "n_eval_report": p.get("n_eval_report")}


def run_metrics(run_dir):
    """One backbone on one label set, or None if nested CV was not run."""
    np_path = os.path.join(run_dir, "nested_results.json")
    if not os.path.isfile(np_path):
        return None
    nr = json.load(open(np_path))
    fam = nr["families"]
    stages = [s for s in fam if s not in ("raw", "embedding")]
    out = {
        "n": nr["protocol"]["n"],
        "raw": {"r2": fam["raw"]["r2_mean"], "sd": fam["raw"]["bootstrap_sd"],
                "choice": _choice(fam["raw"]["chosen"], with_arm=False)},
        "emb": {"r2": fam["embedding"]["r2_mean"], "sd": fam["embedding"]["bootstrap_sd"],
                "choice": _choice(fam["embedding"]["chosen"])},
        "emb_minus_raw": nr["embedding_minus_raw"],
        "blocks": {s: {"r2": fam[s]["r2_mean"], "sd": fam[s]["bootstrap_sd"]} for s in stages},
        "raw_fixed": nr["raw_fixed_recipes"],
    }
    rp = os.path.join(run_dir, "label_probe_results.json")
    if os.path.isfile(rp):
        canary = json.load(open(rp)).get("canary", {}).get("1", {})
        out["canary"] = {"n": len(canary), "passed": sum(1 for v in canary.values() if v["passed"])}
    out["panel"] = _panel(os.path.join(run_dir, "recipe_panel.json"))
    return out


def _rank(eval_dir, ds, tags):
    """Backbones ranked by nested embedding R² on one label set, with the
    paired gap between each consecutive pair."""
    oof = {}
    for t in tags:
        p = os.path.join(_run_dir(eval_dir, t, ds), "nested_oof.npz")
        if os.path.isfile(p):
            oof[t] = np.load(p)
    if len(oof) < 2:
        return None
    y = next(iter(oof.values()))["y"].astype(np.float64)
    for t, d in oof.items():
        assert np.array_equal(d["y"], next(iter(oof.values()))["y"]), f"{t}: labels differ on {ds}"
    score = {t: float(np.mean([1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
                               for p in d["embedding"]])) for t, d in oof.items()}
    order = sorted(score, key=lambda t: -score[t])
    gaps = [paired_delta(y, oof[a]["embedding"], oof[b]["embedding"])
            for a, b in zip(order, order[1:])]
    return {"order": order, "gaps": gaps}


def discover_label_sets(eval_dir, tag):
    base = os.path.join(eval_dir, f"label_probe_regression_{tag}", "label_probe")
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(base, "dataset*")))


def collect(eval_dir, backbones=BACKBONES):
    tags = [b["tag"] for b in backbones]
    sets = discover_label_sets(eval_dir, tags[0])
    everything = sets + [MERGED, LD]
    runs = {t: {ds: run_metrics(_run_dir(eval_dir, t, ds)) for ds in everything} for t in tags}
    sizes = {}
    for ds in everything:
        bank = os.path.join(_run_dir(eval_dir, tags[0], ds), "bank.npz")
        if os.path.isfile(bank):
            sizes[ds] = int(np.load(bank)["y"].shape[0])
    ranking = {ds: _rank(eval_dir, ds, tags) for ds in everything
               if any(runs[t][ds] for t in tags)}
    return {"backbones": backbones, "label_sets": sets, "merged": MERGED, "reference": LD,
            "sizes": sizes, "runs": runs, "ranking": ranking,
            "stage_names": {s: stage_display_name(s) for s in
                            ["fe", "extract_features"] + [f"layer{i}" for i in range(13)]}}
