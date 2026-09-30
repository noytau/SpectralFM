"""
Dimensionality-reduction comparison for the embedding stages (step 6).

Motivated by review: PLS-64 was both the largest k swept in step 4 AND the
winner, which is the signature of an un-found optimum -- so extend k. Plus:
would L1 selection, an MLP head, an autoencoder or UMAP do better?

The organising axis is SUPERVISED vs UNSUPERVISED, not linear vs nonlinear.
Measured in step 4: at matched k, PLS beats PCA 2-6x (3-comp, k~5: 0.389 vs
0.066). The label direction is not in the high-variance directions, so any
reducer optimising variance or reconstruction inherits that loss.

  supervised    pls_k, lasso, elasticnet, mlp_head
  unsupervised  pca_k  (kept only as the reference control)

Considered and dropped by agreement rather than run:
  * plain autoencoder -- optimises reconstruction, the same objective family
    as PCA which is already measured failing here. This project has in effect
    run that experiment: TASKS.md T6 found reconstruction-trained backbones
    were the ones scoring ~0 on label regression.
  * UMAP -- deliberately distorts global structure and its transform on unseen
    data is approximate; wrong trade for regression.

Reduced-rank regression (RRR) was raised and does NOT apply here. RRR minimises
||Y - XB||^2 s.t. rank(B) <= k, but parameter_0 is scalar: Y is n x 1, so B is
p x 1 and rank(B) <= 1 unconditionally -- the constraint is vacuous and RRR
collapses to OLS/ridge. It would become the right tool if the upstream
pipeline's other parameters (the name "parameter_0" implies more exist) were
recovered, making Y multivariate.

Lasso/ElasticNet test something PLS does not: PLS builds dense combinations of
every dimension, L1 SELECTS a sparse subset. That asks directly whether the
label signal is concentrated in a few embedding dimensions or spread across all
of them -- and it names which ones.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import glob, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from . import cache as cachemod
from . import features as feat
from .protocol import r2
from .regressors import make_regressor

PLS_K = (2, 5, 16, 32, 64, 96, 128, 192, 256, 384)
PCA_K = (16, 64, 128, 256)
MLP_HIDDEN = ((64,), (256,), (256, 128))
SPARSE_KINDS = ("lasso", "elasticnet")

_NNZ: list[int] = []      # nonzero coefficient counts -- the sparsity answer


def _fit_mlp(Xtr, ytr, hidden, epochs=200, seed=42, device="cpu"):
    """Supervised MLP head on the frozen embedding (the H4 probe deferred in
    plan.md). Early-stopped on an inner split so it cannot just memorise."""
    import torch, torch.nn as nn
    torch.manual_seed(seed); torch.set_num_threads(2)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym, ys = ytr.mean(), ytr.std() + 1e-8
    X = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=device)
    Y = torch.tensor((ytr - ym) / ys, dtype=torch.float32, device=device).view(-1, 1)
    ntr = int(len(X) * 0.85)
    layers, d = [], Xtr.shape[1]
    for hh in hidden:
        layers += [nn.Linear(d, hh), nn.GELU(), nn.Dropout(0.1)]; d = hh
    layers += [nn.Linear(d, 1)]
    net = nn.Sequential(*layers).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, bstate, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(ntr, device=device)
        for i in range(0, ntr, 256):
            idx = perm[i:i + 256]
            loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = ((net(X[ntr:]) - Y[ntr:]) ** 2).mean().item()
        if v < best - 1e-4:
            best, bad = v, 0
            bstate = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 25:
                break
    if bstate:
        net.load_state_dict(bstate)
    net.eval()

    def pred(Xe):
        with torch.no_grad():
            z = net(torch.tensor((Xe - mu) / sd, dtype=torch.float32, device=device))
        return (z.cpu().numpy().ravel() * ys) + ym
    return pred


def eval_reducer(kind, k, X, y, n_folds=5, seed=42, device="cpu", hidden=None):
    """Out-of-fold R2. Every supervised reducer is fitted INSIDE the training
    fold only -- fitting PLS/Lasso/the MLP on all data would leak labels."""
    cv = KFold(n_folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y))
    for tr, te in cv.split(X):
        Xtr, Xte, ytr = X[tr], X[te], y[tr]
        with threadpool_limits(limits=2):
            if kind in SPARSE_KINDS:
                from sklearn.linear_model import LassoCV, ElasticNetCV
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler().fit(Xtr)
                m = (LassoCV(n_alphas=40, cv=3, random_state=seed, max_iter=5000)
                     if kind == "lasso" else
                     ElasticNetCV(l1_ratio=[.2, .5, .8, .95], n_alphas=30, cv=3,
                                  random_state=seed, max_iter=5000))
                m.fit(sc.transform(Xtr), ytr)
                pred[te] = m.predict(sc.transform(Xte))
                _NNZ.append(int(np.sum(m.coef_ != 0)))
                continue
            if kind == "pls":
                kk = max(1, min(k, Xtr.shape[0] - 1, Xtr.shape[1]))
                m = PLSRegression(n_components=kk).fit(Xtr, ytr)
                pred[te] = np.asarray(m.predict(Xte)).ravel()
                continue
            if kind == "mlp_head":
                pred[te] = _fit_mlp(Xtr, ytr, hidden, seed=seed, device=device)(Xte)
                continue
            if kind == "pca":
                p = PCA(n_components=min(k, *Xtr.shape), random_state=seed).fit(Xtr)
                Ztr, Zte = p.transform(Xtr), p.transform(Xte)
            else:
                raise ValueError(kind)
            reg = make_regressor("ridgecv", seed=seed).fit(Ztr, ytr)
            pred[te] = np.asarray(reg.predict(Zte)).ravel()
    return r2(y, pred)


def canary(out_dir, comps=(1, 3), stages=("transformer",), seed=42,
           kinds=("pls", "pca", "lasso", "mlp_head")):
    """
    Shuffled-label leak canary.

    Permute y, then run the identical evaluation path. With labels destroyed,
    an honest pipeline MUST score R2 ~ 0 (slightly negative out-of-fold).
    Anything meaningfully above 0 means label information is reaching the model
    through a path other than the training labels -- which is exactly the bug
    that inflated PLS by +0.09 here (the supervised projection was fitted on
    all of y, including the test folds, before cross-validation).

    This is cheap and catches a whole class of leakage that eyeballing cannot.
    Run it whenever a supervised reducer or transform enters the pipeline.
    """
    cdir = os.path.join(out_dir, "_cache")
    files = sorted(glob.glob(os.path.join(cdir, "reps_*.npz")), key=os.path.getsize, reverse=True)
    key = os.path.basename(files[0])[len("reps_"):-len(".npz")]
    reps, y, meta = cachemod.load_reps(cdir, key)
    rng = np.random.default_rng(seed)
    y_shuf = rng.permutation(y)

    print(f"{'config':<34}{'real y':>10}{'shuffled y':>13}  verdict", flush=True)
    out = {}
    for c in comps:
        ci = [feat.UNIQUE_COMPS.index(x) for x in feat.COMP_LADDER[c]]
        for st in stages:
            X = feat.make_wide(reps[st], ci)
            for kind in kinds:
                k = 64 if kind in ("pls", "pca") else None
                h = (256,) if kind == "mlp_head" else None
                real = eval_reducer(kind, k, X, y, seed=seed, hidden=h)
                shuf = eval_reducer(kind, k, X, y_shuf, seed=seed, hidden=h)
                ok = shuf < 0.02
                nm = f"{c}c {st} {kind}{'_' + str(k) if k else ''}"
                out[nm] = {"real": real, "shuffled": shuf, "pass": bool(ok)}
                print(f"{nm:<34}{real:>+10.4f}{shuf:>+13.4f}  "
                      f"{'PASS' if ok else '*** LEAK ***'}", flush=True)
    p = os.path.join(out_dir, "step6_leak_canary.json")
    json.dump(out, open(p, "w"), indent=2, default=str)
    print("wrote", p, flush=True)
    return out


def run(out_dir, comps=(1, 3), stages=("transformer",), device="cpu",
        kinds=("pls", "pca", "lasso", "elasticnet", "mlp_head"), seed=42, tag=""):
    cdir = os.path.join(out_dir, "_cache")
    files = sorted(glob.glob(os.path.join(cdir, "reps_*.npz")), key=os.path.getsize, reverse=True)
    key = os.path.basename(files[0])[len("reps_"):-len(".npz")]
    reps, y, meta = cachemod.load_reps(cdir, key)
    res, t0 = {}, time.time()
    grids = {"pls": PLS_K, "pca": PCA_K}
    for c in comps:
        ci = [feat.UNIQUE_COMPS.index(x) for x in feat.COMP_LADDER[c]]
        for st in stages:
            X = feat.make_wide(reps[st], ci)
            print(f"[step6] {c}-comp {st}  X={X.shape}", flush=True)
            for kind in kinds:
                if kind in SPARSE_KINDS:
                    _NNZ.clear()
                    v = eval_reducer(kind, None, X, y, seed=seed, device=device)
                    nnz = int(np.mean(_NNZ)) if _NNZ else -1
                    res[f"{c}|{st}|{kind}"] = v
                    res[f"{c}|{st}|{kind}__nnz"] = nnz
                    print(f"  {c}c {st:<12} {kind:<16} R2={v:+.4f}  "
                          f"nonzero={nnz}/{X.shape[1]} [{time.time()-t0:.0f}s]", flush=True)
                elif kind == "mlp_head":
                    for h in MLP_HIDDEN:
                        nm = f"mlp_{'x'.join(map(str, h))}"
                        v = eval_reducer(kind, None, X, y, seed=seed, device=device, hidden=h)
                        res[f"{c}|{st}|{nm}"] = v
                        print(f"  {c}c {st:<12} {nm:<16} R2={v:+.4f} "
                              f"[{time.time()-t0:.0f}s]", flush=True)
                else:
                    for k in grids[kind]:
                        if k >= min(X.shape):
                            continue
                        v = eval_reducer(kind, k, X, y, seed=seed, device=device)
                        res[f"{c}|{st}|{kind}_{k}"] = v
                        print(f"  {c}c {st:<12} {kind}_{k:<11} R2={v:+.4f} "
                              f"[{time.time()-t0:.0f}s]", flush=True)
                json.dump(res, open(os.path.join(out_dir, f"step6_reducers{tag}.json"), "w"),
                          indent=2, default=str)
    print("wrote", os.path.join(out_dir, f"step6_reducers{tag}.json"), flush=True)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--comps", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--stages", nargs="+", default=["transformer"])
    ap.add_argument("--kinds", nargs="+",
                    default=["pls", "pca", "lasso", "elasticnet", "mlp_head"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="")
    ap.add_argument("--canary", action="store_true",
                    help="shuffled-label leak check; every kind must score ~0")
    a = ap.parse_args()
    if a.canary:
        canary(a.out_dir, tuple(a.comps), tuple(a.stages))
        raise SystemExit(0)
    run(a.out_dir, tuple(a.comps), tuple(a.stages), a.device, tuple(a.kinds), tag=a.tag)
