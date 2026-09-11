import time, sys
from eval.label_probe import crossover as cx, geometry as geo, readouts as ro, screen as scr, study6

S = "/mnt5/home/hadar/nova/SpectralFM-few-shot/code/eval_outputs/label_probe/step6/bank/bank.npz"
print("loading bank...", flush=True)
t_load0 = time.time()
bank, raw, z, y, meta = ro.load_bank_cache(S)
print(f"bank loaded in {time.time()-t_load0:.1f}s", flush=True)

split = scr.make_split(len(y), seed=42)
cache = {}
rd = cx.CANDIDATES[3][0]
print(f"materializing readout {rd.label()} for 3-comp...", flush=True)
t0 = time.time()
X = study6.materialize(rd, bank, raw, z, [0, 1, 2], cache)
t1 = time.time()
print(f"materialize {t1-t0:.1f}s (once per readout, reused across rungs)", flush=True)

for n in (20, 2000):
    draws = {n: scr.draw_indices(split["pool"], n, cx.DRAWS_BY_N[n], seed=42)}
    t2 = time.time()
    scr.score_readout(X, y, split, cx.panel_for_n(n), (n,), draws, "eval_a",
                      probe_factory=cx.make_probe_n)
    dt = time.time() - t2
    print(f"n={n:<5} panel={len(cx.panel_for_n(n))} probes  score {dt:6.1f}s", flush=True)

print("est. total = 11 readouts x 3 comps x (materialize + sum over 7 rungs x 2 eval sets)", flush=True)
print("DONE", flush=True)
