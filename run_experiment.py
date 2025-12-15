import os
from xml.etree.ElementPath import prepare_self

import pandas as pd
import argparse
import wandb
import sys
import os
from model_loader import train_model

from data_parser import run_data_parser
from args_parser import ArgsParser
from model_loader import *

import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# fixme remove
import sys
sys.path.append("/mnt5/noy/fairseq") # fixme change to your fairseq path

def adapt_data2vec_cfg(cfg_dict): # fixme move to model_loader.py / why needed?
    """
    Converts the checkpoint-style dict config into a Data2VecAudioConfig-like object.
    Keeps compatibility with the original fairseq model.
    """
    from omegaconf import OmegaConf

    # helper: safely get keys with fallback
    def get(d, key, default=None):
        return d[key] if isinstance(d, dict) and key in d else getattr(d, key, default)

    # extract nested fields
    audio_mod = get(cfg_dict, "modalities", {}).get("audio", {})
    embed_dim = get(cfg_dict, "encoder_embed_dim", get(cfg_dict, "embed_dim", 768))

    adapted = {
        "encoder_embed_dim": embed_dim,
        "feature_encoder_spec": audio_mod.get("feature_encoder_spec", "[(512,10,5)]"),
        "extractor_mode": audio_mod.get("extractor_mode", "layer_norm"),
        "conv_bias": audio_mod.get("conv_bias", False),
        "dropout_input": get(cfg_dict, "dropout_input", 0.0),
        "dropout_features": get(cfg_dict, "dropout_features", 0.0),
        "average_top_k_layers": get(cfg_dict, "average_top_k_layers", 8),
        "loss_beta": get(cfg_dict, "loss_beta", 0.0),
        "loss_scale": get(cfg_dict, "loss_scale", None),
        "feature_grad_mult": get(cfg_dict, "feature_grad_mult", 1.0),
        "mask_prob": audio_mod.get("mask_prob", 0.5),
        "mask_length": audio_mod.get("mask_length", 5),
    }

    # merge the rest of cfg keys (to preserve compatibility)
    merged = {**cfg_dict, **adapted}

    # defaults for fairseq encoder
    if "pos_conv_depth" not in merged or merged["pos_conv_depth"] is None:
        merged["pos_conv_depth"] = 1
    if "pos_conv_kernel" not in merged or merged["pos_conv_kernel"] is None:
        merged["pos_conv_kernel"] = 3
    if not isinstance(merged.get("conv_pos"), int):
        merged["conv_pos"] = 128
    if not isinstance(merged.get("conv_pos_groups"), int):
        merged["conv_pos_groups"] = 16

    # return as OmegaConf object for fairseq
    return OmegaConf.create(merged)

# fixme remove in the end
class AudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, max_len=None):
        self.root_dir = root_dir
        self.wav_files = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".wav")
        ]
        self.sample_rate = sample_rate
        self.max_len = max_len

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]
        wav, sr = torchaudio.load(wav_path)

        # resample if needed
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # trim or pad to max_len
        if self.max_len:
            wav = wav[:, :self.max_len]
            if wav.shape[1] < self.max_len:
                pad_len = self.max_len - wav.shape[1]
                wav = torch.nn.functional.pad(wav, (0, pad_len))

        return wav, wav_path

def load_wav_dataloader(directory, batch_size=4, sample_rate=16000, max_len=None, num_workers=2):
    dataset = AudioDataset(directory, sample_rate, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def extract_embeddings_from_loader(model, dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.eval()

    all_features = []
    all_paths = []

    with torch.no_grad():
        for wav_batch, paths in tqdm(dataloader, desc="Extracting embeddings"):
            wav_batch = wav_batch.to(device)
            padding_mask = torch.zeros(wav_batch.shape, dtype=torch.bool, device=device)
            out = model.extract_features(wav_batch, padding_mask)
            features = out["x"].mean(dim=1).cpu()
            all_features.append(features)
            all_paths.extend(paths)

    embeddings = torch.cat(all_features, dim=0)
    return embeddings, all_paths


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

    def run_inference(self ):
        from examples.data2vec.models.data2vec_audio import Data2VecAudioModel

        from omegaconf import OmegaConf

        model_path = "/mnt5/noy/fairseq/base_libri.pt"
        state = torch.load(model_path, map_location="cpu")

        cfg = create_data2vec_audio_config()
        model = Data2VecAudioModel.build_model(cfg)
        model.load_state_dict(state["model"], strict=False)

        dataloader = load_wav_dataloader("/mnt5/noy/fairseq/LibriSpeech/wav/dev-clean/1272/128104/", batch_size=8)
        embeddings, paths = extract_embeddings_from_loader(model, dataloader)
        print("Embeddings shape:", embeddings.shape)

    def run_inference_one(self):
        import sys
        sys.path.append("/mnt5/noy/fairseq")  # fixme change to your fairseq path
        from transformers import Data2VecAudioConfig
        from examples.data2vec.models.data2vec_audio import Data2VecAudioModel
        from examples.data2vec.models.modalities.audio import AudioEncoder
        from fairseq.data.dictionary import Dictionary
        from omegaconf import OmegaConf
        import torch, torchaudio
        #df = self.get_libri_audio_data()
        #model, feature_extractor, optimizer, device = load_fairseq_data2vec_model(self.args)
        model_path = "/mnt5/noy/fairseq/base_libri.pt"

        torch.serialization.add_safe_globals([Dictionary])

        state = torch.load(model_path, map_location="cpu")

        # fixme - base_cfg = OmegaConf.load("/mnt5/noy/fairseq/examples/data2vec/config/audio/pretraining/base_librispeech.yaml")
        cfg = create_data2vec_audio_config()

        model = Data2VecAudioModel.build_model(cfg)
        model.load_state_dict(state["model"], strict=False)
        model.eval()
        # load 1 sample of audio
        wav, sr = torchaudio.load("/mnt5/noy/fairseq/LibriSpeech/wav/dev-clean/1272/128104/1272-128104-0004.wav")

        with torch.no_grad():
            wav = wav.unsqueeze(0) if wav.ndim == 1 else wav  # [1, T]
            padding_mask = torch.zeros(wav.shape, dtype=torch.bool, device=wav.device)

            out = model.extract_features(wav, padding_mask)
            features = out["x"].mean(dim=1)

        print("Feature shape:", features.shape)

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
                            default='/mnt5/noy/fairseq/outputs/2025-12-01/08-13-50/checkpoints/checkpoint_last.pt',
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

