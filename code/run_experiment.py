import os
from xml.etree.ElementPath import prepare_self

import pandas as pd
import argparse
import sys
import os
from model_loader import train_model

from data_parser import run_data_parser
from args_parser import ArgsParser
from model_loader import *

import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# fixme remove




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
        print("NOY")
        df = self.prepare_data()
        print("NOY: Data prepared with shape:", df.shape)
        # run training
        model_path = train_model(df, self.args)
        print(f"[+] Model training finished, model saved at: {model_path}")

    def get_libri_audio_data(self):
        convert_flac_to_wav(input_root="/mnt5/noy/fairseq/LibriSpeech/", output_root="/mnt5/noy/fairseq/LibriSpeech/wav/")


if __name__ == "__main__":
    # Parse command line arguments in main
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="/mnt5/noy/code/logs/run_masking.csv")
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--arch", type=str)
    parser.add_argument("--masking_type", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "inference"], default="eval")

    if parser.parse_args().mode == 'eval':
        parser.add_argument("--eval_method", type=str, choices=["signal_completion", "noise_robustness", "compare_embeddings", "classifier_head"], default="signal_completion")
        parser.add_argument("--test_method", type=str, choices=["index_out_of_distribution", "index_in_distribution_stack_holdout", "test_in_distribution_partial_stack"], default="index_in_distribution_stack_holdout")
        parser.add_argument('--model_path', type=str,
                            default='/mnt5/noy/fairseq/outputs/2025-12-01/08-13-50/checkpoints/checkpoint_best.pt',
                            help='Path to saved model')
        #                            default='/mnt5/noy/code/weights/experiment-mask=0.25-epoch=1_batch=32_loss_fn=mse_datalen=1000000_model_after_training.pt',

    args = parser.parse_args()
    # Use the modular parser
    args_parser = ArgsParser(args.csv_path)
    full_args = args_parser.parse(args.run_id, args)
    if args.mode == "train":
        runner = ExperimentRunner(full_args)
        runner.run_experiment()
    elif args.mode == "eval":
        from evaluate import EvalExperiment  # fixme move import to avoid circular import
        full_args.model, feature_extractor, optimizer, device = load_trained_model(full_args)  # Implement this function to load the trained model
        evaluator = EvalExperiment(full_args)
        evaluator.run_evaluation()
    elif args.mode == "inference": # run inference on pre-trained data2vec model
        runner = ExperimentRunner(full_args) # fixme implement inference mode?
        runner.run_inference()

