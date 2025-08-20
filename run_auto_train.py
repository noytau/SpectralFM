import os
import pandas as pd
import argparse
import wandb

class ExperimentRunner:
    def __init__(self, csv_path, script_path="run_auto_train.py"):
        """
        Initialize the ExperimentRunner with CSV and script paths.
        """
        self.csv_path = csv_path
        self.script_path = script_path
        self.df = self.load_runs_csv()

    def load_runs_csv(self):
        """
        Load the experiment runs table from the CSV.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at: {self.csv_path}")
        return pd.read_csv(self.csv_path)

    def generate_run_command(self, run_id):
        """
        Generate a command to run the training script with parameters from the CSV.
        """
        row = self.df[self.df["Run ID"] == run_id].iloc[0]

        command = (
            f"/home/noy/.virtualenvs/SpectralFM/bin/python {self.script_path} "
            f"--learning_rate {row['LR']} "
            f"--mask_ratio {row['Mask Ratio']} "
            f"--batch_size {row['Batch Size']} "
            f"--epoch {row['Epochs']} "
            f"--loss_function {row['Loss Function']} "
            # f"--masking_technique {row['Masking Technique']} " fixme add in the future, only random indexes for now 
            f"--run_id {row['Run ID']} "
            f"--test_dir=medium")  # for now, run only 1m sample
        return command

    def run_experiment_by_id(self, run_id):
        """
        Execute the training command for a given run ID.
        """
        command = self.generate_run_command(run_id)
        print(f"Running command:\n{command}")
        try:
            os.system(command)
        except Exception as e:
            print("Run failed due to:", e)
            import traceback
            traceback.print_exc()
            wandb.alert(title="Run failed", text=str(e))
            raise
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments from CSV by run_id")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--run_id", type=int, required=True, help="Run ID to execute")
    parser.add_argument("--script_path", type=str, default="/mnt5/noy/code/main.py", help="Training script path") # fixme change to train

    args = parser.parse_args()

    runner = ExperimentRunner(csv_path=args.csv_path, script_path=args.script_path)
    runner.run_experiment_by_id(args.run_id)