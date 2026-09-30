import time
import numpy as np
from eval.label_probe import geometry as geo, readouts as ro, screen as scr, study6

BANK = "/mnt5/home/hadar/nova/SpectralFM-few-shot/code/eval_outputs/label_probe/step6/bank/bank.npz"
bank, raw, z, y, meta = ro.load_bank_cache(BANK)
split = scr.make_split(len(y), seed=42)
draws = {20: scr.draw_indices(split["pool"], 20, 30, seed=42),
         50: scr.draw_indices(split["pool"], 50, 30, seed=42)}
cache = {}

# Expensive corner readout, full panel
rd = geo.Readout("emb", "layer12", "mean", "whiten")
t0 = time.time(); X = study6.materialize(rd, bank, raw, z, [0, 1, 2], cache); t1 = time.time()
print(f"materialize {t1-t0:.1f}s", flush=True)

t1b = time.time()
scr.score_readout(X, y, split, geo.SCREEN_PANEL, (20, 50), draws, "eval_a")
t2 = time.time()
print(f"score (full panel incl gp_rbf) {t2-t1b:.1f}s", flush=True)

t3 = time.time(); scr.readout_diagnostics(X, y); t4 = time.time()
print(f"diagnostics {t4-t3:.1f}s", flush=True)

per_full = (t1 - t0) + (t2 - t1b) + (t4 - t3)
print(f"-> {per_full:.1f}s/readout (full panel)", flush=True)
print(f"screen estimate (corrected 152/comp x3): {per_full * 152 * 3 / 3600:.2f} h", flush=True)

# Isolate gp_rbf vs rest of panel
panel_no_gp = {k: v for k, v in geo.SCREEN_PANEL.items() if k != "gp_rbf"}
t5 = time.time()
scr.score_readout(X, y, split, panel_no_gp, (20, 50), draws, "eval_a")
t6 = time.time()
print(f"score (panel WITHOUT gp_rbf) {t6-t5:.1f}s", flush=True)

gp_only = {"gp_rbf": geo.SCREEN_PANEL["gp_rbf"]}
t7 = time.time()
scr.score_readout(X, y, split, gp_only, (20, 50), draws, "eval_a")
t8 = time.time()
print(f"score (gp_rbf ONLY) {t8-t7:.1f}s", flush=True)

per_no_gp = (t1 - t0) + (t6 - t5) + (t4 - t3)
print(f"-> {per_no_gp:.1f}s/readout (panel WITHOUT gp_rbf, materialize+score_nogp+diagnostics)", flush=True)
print(f"screen estimate WITHOUT gp_rbf (corrected 152/comp x3): {per_no_gp * 152 * 3 / 3600:.2f} h", flush=True)
print("DONE", flush=True)
