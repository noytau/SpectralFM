import argparse
import pandas as pd

PARAMS = {
    "run_id":        {"column": "Run ID",         "default": None},
    "test_dir":      {"column": "Test Dir",       "default": "medium"},
    "arch":          {"column": "FE Architecture","default": "conv1d"},
    "mask_ratio":    {"column": "Mask Ratio",     "default": 0.15},
    "masking_type":  {"column": "Masking Type",   "default": "span"},
    "loss_function": {"column": "Loss Function",  "default": "mse"},
    "learning_rate": {"column": "LR",             "default": 1e-4},
    "batch_size":    {"column": "Batch Size",     "default": 32},
    "epoch":         {"column": "Epochs",         "default": 1},
    "model_path":    {"column": "Model Path",     "default": None},
    "output_path":   {"column": "Output Path",    "default": "/mnt5/noy/code/outputs"},
    "mode":          {"column": "Mode",           "default": "train"},
    "eval_method":   {"column": "Eval Method",    "default": "signal_completion"},
    "test_method":   {"column": "Test Method",    "default": "index_in_distribution_stack_holdout"},
    "stack_method": {"column": "Stack Method",    "default": "index"},
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
        def get_arg(key):
            col = PARAMS[key]["column"]
            default = PARAMS[key]["default"]
            cli_val = getattr(cli_args, key, None) if cli_args else None
            if cli_val is not None:
                return cli_val
            elif col in row and pd.notna(row[col]):
                return row[col]
            else:
                return default

        if getattr(cli_args, "debug", False):
            args.test_dir = "small"
        else:
            args.test_dir = get_arg("test_dir")

        args.arch = get_arg("arch")
        args.masking_type = get_arg("masking_type")
        args.mask_ratio = get_arg("mask_ratio")
        args.loss_function = get_arg("loss_function")
        args.learning_rate = float(get_arg("learning_rate"))
        args.batch_size = int(get_arg("batch_size"))
        args.epoch = int(get_arg("epoch"))
        args.model_path = get_arg("model_path")
        args.model_name = args.model_path.split('/')[-1].strip(".pt").strip("\'") if args.model_path else None
        args.output_path = get_arg("output_path")
        args.mode = get_arg("mode")
        args.eval_method = get_arg("eval_method")
        args.test_method = get_arg("test_method")
        args.stack_method = get_arg("stack_method")

        return args