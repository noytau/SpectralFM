import os
from xml.etree.ElementPath import prepare_self

import pandas as pd
import argparse
import wandb
from model_loader import train_model

from data_parser import run_data_parser
from args_parser import ArgsParser

class ExperimentRunner:
    def __init__(self, experiment_args):
        self.args = experiment_args

    def get_experiment_args(self):
        # Use the modular parser
        args_parser = ArgsParser(args.csv_path)
        full_args = args_parser.parse(args.run_id, args)

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

    def prepare_data(self):
        samples_path = self.get_samples_path(self.args.test_dir)
        # parse data
        self.df = run_data_parser(samples_path)
        return self.df

    def run_experiment(self):
        df = self.prepare_data()
        # run training
        model_path = train_model(df, self.args)
        print(f"[+] Model training finished, model saved at: {model_path}")


if __name__ == "__main__":
    # Parse command line arguments in main
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="/mnt5/noy/code/logs/run_masking.csv")
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--arch", type=str)
    parser.add_argument("--masking_type", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="eval")

    if parser.parse_args().mode == 'eval':
        parser.add_argument("--eval_method", type=str, choices=["signal_completion", "noise_robustness", "compare_embeddings", "classifier_head"], default="signal_completion")
        parser.add_argument("--test_method", type=str, choices=["index_out_of_distribution", "index_in_distribution_stack_holdout", "test_in_distribution_partial_stack"], default="index_in_distribution_stack_holdout")

    args = parser.parse_args()
    # Use the modular parser
    args_parser = ArgsParser(args.csv_path)
    full_args = args_parser.parse(args.run_id, args)

    if args.mode == "train":
        runner = ExperimentRunner(full_args)
        runner.run_experiment()
    elif args.mode == "eval":
        from evaluate import EvalExperiment  # fixme move import to avoid circular import

        evaluator = EvalExperiment(full_args)
        evaluator.run_evaluation()

