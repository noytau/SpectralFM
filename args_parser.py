import argparse
import pandas as pd

PARAMS = {
    "run_id":        {"column": "Run ID",         "default": None},
    "test_dir":      {"column": "Test Dir",       "default": "medium"},
    "arch":          {"column": "FE Architecture","default": "conv1d"},
    "masking_type":  {"column": "Masking Type",   "default": "span"},
    "loss_function": {"column": "Loss Function",  "default": "mse"},
    "learning_rate": {"column": "LR",             "default": 1e-4},
    "batch_size":    {"column": "Batch Size",     "default": 32},
    "epoch":         {"column": "Epochs",         "default": 1},
}

class ArgsParser:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path) if csv_path else None

    def get_row(self, run_id):
        if self.df is not None:
            return self.df[self.df["Run ID"] == run_id].iloc[0]
        return {}

    def parse(self, run_id, cli_args=None):
        row = self.get_row(run_id)
        args = argparse.Namespace()
        args.run_id = run_id

        # Use CLI first, fallback to CSV, then default
        def get_arg(key, cli_val=None):
            col = PARAMS[key]["column"]
            default = PARAMS[key]["default"]

            if cli_val is not None:
                return cli_val
            elif col in row and pd.notna(row[col]):
                return row[col]
            else:
                return default

        args.test_dir = "small" if getattr(cli_args, "debug", False) else get_arg("Test Dir", getattr(cli_args, "test_dir", None))
        args.arch = get_arg("arch", getattr(cli_args, "arch", None))
        args.masking_type = get_arg("masking_type", getattr(cli_args, "masking_type", None))
        args.loss_function = get_arg("loss_function", getattr(cli_args, "loss_function", None))
        args.learning_rate = float(get_arg("learning_rate", getattr(cli_args, "learning_rate", None)))
        args.batch_size = int(get_arg("batch_size"))
        args.epoch = int(get_arg("epoch"))

        return args