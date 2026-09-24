import sys
from eval.label_probe import report6, study6

BANK = "/mnt5/home/hadar/nova/SpectralFM-few-shot/code/eval_outputs/label_probe/step6/bank/bank.npz"
OUT = "/mnt5/home/hadar/nova/SpectralFM-few-shot/code/eval_outputs/label_probe/step6/smoke"

print("starting screen", flush=True)
s = study6.run_screen(BANK, OUT, comp_counts=(1,), n_draws=2, n_top_stages=1)
print("screen done", flush=True)
c = study6.run_confirm(BANK, s, OUT, n_draws=3, top_emb=1, top_raw=1,
                        use_tabpfn=False)
print("confirm done", flush=True)
v = study6.verdict(c)
print("VERDICT", v["verdict"], v["n_cells_won"], "/", v["n_cells"], flush=True)
path = report6.write_step6_report(s, c, v, OUT)
print("REPORT_WRITTEN", path, flush=True)
