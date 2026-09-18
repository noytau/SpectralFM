import time
from eval.label_probe import geometry as geo, readouts as ro, screen as scr, study6

BANK = "/mnt5/home/hadar/nova/SpectralFM-few-shot/code/eval_outputs/label_probe/step6/bank/bank.npz"
bank, raw, z, y, meta = ro.load_bank_cache(BANK)
split = scr.make_split(len(y), seed=42)
draws = {20: scr.draw_indices(split["pool"], 20, 30, seed=42),
         50: scr.draw_indices(split["pool"], 50, 30, seed=42)}
cache = {}

rd = geo.Readout("emb", "layer12", "mean", "whiten")
t0 = time.time(); X = study6.materialize(rd, bank, raw, z, [0, 1, 2], cache); t1 = time.time()
print(f"materialize {t1-t0:.1f}s", flush=True)

panel_no_gp = tuple(m for m in geo.SCREEN_PANEL if m != "gp_rbf")
t5 = time.time()
scr.score_readout(X, y, split, panel_no_gp, (20, 50), draws, "eval_a")
t6 = time.time()
print(f"score (panel WITHOUT gp_rbf) {t6-t5:.1f}s", flush=True)

gp_only = ("gp_rbf",)
t7 = time.time()
scr.score_readout(X, y, split, gp_only, (20, 50), draws, "eval_a")
t8 = time.time()
print(f"score (gp_rbf ONLY) {t8-t7:.1f}s", flush=True)
print("DONE", flush=True)
