import os
import pandas as pd
import argparse
import wandb
from model_loader import train_model

from data_parser import run_data_parser
from args_parser import ArgsParser


class ExperimentRunner:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = self.load_runs_csv()

    def load_runs_csv(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at: {self.csv_path}")
        return pd.read_csv(self.csv_path)

    def get_samples_path(self, test_dir):
        base = "/mnt5/noy/nova_samples/"
        return {
            "small": base + "debug_chnl/",
            "medium": base + "one_chnl/",
            "large": base + "full_chnl/"
        }.get(test_dir, base + "debug_chnl/")

    def run_experiment(self, run_id, cli_args=None):
        row = self.df[self.df["Run ID"] == run_id].iloc[0]

        # create an Args-like object for compatibility
        class Args:
            pass
        # Parse args from csv tracking file
        args = Args()
        args.run_id = int(row["Run ID"])
        args.learning_rate = float(row["LR"])
        args.mask_ratio = float(row["Mask Ratio"])
        args.batch_size = int(row["Batch Size"])
        args.epoch = int(row["Epochs"])
        args.loss_function = str(row["Loss Function"])

        # Support args from CLI for development process
        if cli_args != None:
            # running dir - if running in debug mode, use small dir (100K)
            if getattr(cli_args, "debug", False):
                args.test_dir = "small"
            elif "Test Dir" in row and pd.notna(row["Test Dir"]):
                args.test_dir = row["Test Dir"]
            else:
                args.test_dir = "medium"

            if getattr(cli_args, "arch", False):
                args.arch = cli_args.arch
            elif "FE Architecture" in row.index and pd.notna(row["FE Architecture"]):
                args.arch = row["FE Architecture"]
            else:
                args.arch = "conv1d"

            if getattr(cli_args, "masking_type", None):
                args.masking_type = cli_args.masking_type
            elif "Masking Type" in row.index and pd.notna(row["Masking Type"]):
                args.masking_type = row["Masking Type"]
            else: # default type
                args.masking_type = "span"

        samples_path = self.get_samples_path(args.test_dir)
        # parse data
        df = run_data_parser(samples_path)

        # run training
        model_path = train_model(df, args)
        print(f"[+] Model training finished, model saved at: {model_path}")


if __name__ == "__main__":
    # in main:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="/mnt5/noy/code/logs/run_masking.csv")
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--arch", type=str)
    parser.add_argument("--masking_type", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    runner = ExperimentRunner(csv_path=args.csv_path)

    # Use the modular parser
    args_parser = ArgsParser(args.csv_path)
    full_args = args_parser.parse(args.run_id, args)

    runner.run_experiment(args.run_id, full_args)
