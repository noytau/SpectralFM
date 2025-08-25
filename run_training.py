import os
import pandas as pd
import argparse
import wandb
from model_loader import train_model

from data_parser import run_data_parser


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

    def run_experiment(self, run_id):
        row = self.df[self.df["Run ID"] == run_id].iloc[0]

        # create an Args-like object for compatibility
        class Args:
            pass

        args = Args()
        args.run_id = int(row["Run ID"])
        args.learning_rate = float(row["LR"])
        args.mask_ratio = float(row["Mask Ratio"])
        args.batch_size = int(row["Batch Size"])
        args.epoch = int(row["Epochs"])
        args.loss_function = str(row["Loss Function"])
        args.test_dir = "medium"  # or from CSV if you store it

        samples_path = self.get_samples_path(args.test_dir)
        # parse data
        df = run_data_parser(samples_path)

        # run training
        model_path = train_model(df, args)
        print(f"[+] Model training finished, model saved at: {model_path}")

    def generate_run_command(self, run_id, script_path):
        row = self.df[self.df["Run ID"] == run_id].iloc[0]
        command = (
            f"/home/noy/.virtualenvs/SpectralFM/bin/python {script_path} "
            f"--learning_rate {row['LR']} "
            f"--mask_ratio {row['Mask Ratio']} "
            f"--batch_size {row['Batch Size']} "
            f"--epoch {row['Epochs']} "
            f"--loss_function {row['Loss Function']} "
            f"--run_id {row['Run ID']} "
            f"--test_dir=small"
        )
        return command

    def generate_run_args(self, run_id):
        row = self.df[self.df["Run ID"] == run_id].iloc[0]

        class Args:
            pass

        args = Args()
        args.run_id = row["Run ID"]
        args.learning_rate = row["LR"]
        args.mask_ratio = row["Mask Ratio"]
        args.batch_size = row["Batch Size"]
        args.epoch = row["Epochs"]
        args.loss_function = row["Loss Function"]
        args.test_dir = "small"  # or from row["Test Dir"] if available

        return args


    def run_experiment_by_id(self, run_id, script_path):
        command = self.generate_run_command(run_id, script_path)
        print(f"Running command:\n{command}")
        try:
            os.system(command)
        except Exception as e:
            print("Run failed due to:", e)
            wandb.alert(title="Run failed", text=str(e))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="/mnt5/noy/code/logs/runs_tracking.csv")
    parser.add_argument("--run_id", type=int, default=1)
    args = parser.parse_args()

    runner = ExperimentRunner(csv_path=args.csv_path)

    # directly run training logic
    runner.run_experiment(args.run_id)
