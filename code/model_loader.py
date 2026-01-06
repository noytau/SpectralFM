from transformers import Data2VecAudioModel, AutoFeatureExtractor
import torch
import sys
import os

# Add fairseq to path using relative path from this file's location
_FAIRSEQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fairseq")
if _FAIRSEQ_PATH not in sys.path:
    sys.path.insert(0, _FAIRSEQ_PATH)

from fairseq import checkpoint_utils
#from examples.data2vec.models import Data2VecAudioModel
#from examples.data2vec.models.modalities.audio import Data2VecAudioModel
from examples.data2vec.models.data2vec_audio import Data2VecAudioModel
import soundfile
import torch, soundfile as sf
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor, Data2VecAudioModel, AutoConfig, TrainingArguments, Data2VecAudioConfig
import torch
import torch.nn as nn
from datasets import Dataset
import numpy as np
import random
from trainer import SelfSupervisedDataCollator, SelfSupervisedTrainer
from customize_model import *
import wandb
import mlflow
import mlflow.pytorch

def mask_spectrogram(example, mask_ratio=0.15, mask_value=0.0):
    """
    Mask a percentage of random cells within each row in the 'data' field of a Hugging Face dataset example.

    Args:
        example (dict): A sample from the dataset, must include 'data' key.
        mask_ratio (float): Fraction of values to mask per row.
        mask_value (float): Value to use for masking (e.g., 0.0 or np.nan).

    Returns:
        dict: Updated sample with a new 'masked_data' field, and list of masked cell indices as (row, col) tuples.
    """
    data = np.array(example["data"])
    masked = data.copy()
    mask_indices = []

    for i, row in enumerate(data):
        row_indices = random.sample(range(len(row)), int(mask_ratio * len(row)))
        for idx in row_indices:
            masked[i, idx] = mask_value
            mask_indices.append((i, idx))

    return {
        "data": data.tolist(),
        "masked_data": masked.tolist(),
        "mask_indices": mask_indices
    }


# Visualization of masked dataset statistics
import matplotlib.pyplot as plt
import os


def load_custom_data2vec_audio_model(args, model_name="facebook/data2vec-audio-base"):
    model = Data2VecAudioModel.from_pretrained(model_name)
    # fixme add all these to a custom function
    # change to 1 layer feature extractor
    model.feature_extractor = CustomFeatureExtractor(args.arch)
    model.completion_head = CompletionHead(hidden_dim=768, output_dim=245)
    model.config.do_stft_input = True
    # freeze all layers apart from feature extractor
    for param in model.parameters():
        param.requires_grad = False
    for param in model.feature_extractor.parameters():
        param.requires_grad = True
    # Define optimizer for trainable params
    args.optimizer = torch.optim.Adam(model.feature_extractor.parameters(), args.learning_rate)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model = model.to(args.device) # fixme what am I doing here
    return model.to(args.device), model.feature_extractor, args.optimizer, args.device

def load_fairseq_checkpoint(checkpoint_path, device=None):
    """
    Load a fairseq data2vec checkpoint from the outputs directory.
    
    Args:
        checkpoint_path: Path to checkpoint file (e.g., checkpoint_best.pt)
        device: Device to load model on (default: auto-detect GPU/CPU)
    
    Returns:
        model: The loaded Data2VecAudioModel
        cfg: The model configuration
        checkpoint_info: Dict with training info (step, epoch, etc.)
    """
    from fairseq import checkpoint_utils
    from omegaconf import OmegaConf, open_dict
    # Import fairseq model explicitly (not the transformers version)
    from examples.data2vec.models.data2vec_audio import Data2VecAudioModel as FairseqData2VecAudioModel
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[+] Loading fairseq checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path, arg_overrides={})
    
    # Extract configuration
    cfg = state["cfg"]
    model_cfg = cfg["model"]
    
    # Handle missing config keys for older checkpoints
    default_keys = {
        "model_path": None,
        "skip_pretrained_weights": True,
        "train_only_fe": False,
    }
    
    with open_dict(model_cfg):
        for key, default_val in default_keys.items():
            if key not in model_cfg:
                model_cfg[key] = default_val
                print(f"[+] Added missing config key: {key}={default_val}")
    
    # Build model from config using fairseq model class
    model = FairseqData2VecAudioModel.build_model(model_cfg)
    
    # Load model weights
    model_state = state.get("model", {})
    
    # Remove EMA keys if present
    if "_ema" in model_state:
        del model_state["_ema"]
    
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    
    if missing:
        print(f"[!] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[!] Unexpected keys: {len(unexpected)}")
    
    model = model.to(device)
    model.eval()
    
    checkpoint_info = {
        "num_updates": state.get("optimizer_history", [{}])[-1].get("num_updates", 0) if state.get("optimizer_history") else 0,
        "epoch": state.get("epoch", 0),
        "best_loss": state.get("best", None),
        "cfg": cfg,
    }
    
    print(f"[+] Model loaded successfully!")
    print(f"    - Epoch: {checkpoint_info['epoch']}")
    print(f"    - Updates: {checkpoint_info['num_updates']}")
    
    return model, model_cfg, checkpoint_info


def load_fairseq_model_for_evaluation(model_path, device=None):
    """
    Load a fairseq checkpoint for evaluation.
    
    Args:
        model_path: Path to checkpoint_best.pt or any checkpoint file
        device: Target device
        
    Returns:
        model: Model in eval mode
        cfg: Full configuration dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, model_cfg, checkpoint_info = load_fairseq_checkpoint(model_path, device)
    model.eval()
    
    return model, checkpoint_info["cfg"]


def load_fairseq_data2vec_model(args, model_path="/mnt5/noy/fairseq/base_libri_960h.pt"):
    """Legacy function - loads fairseq checkpoint with args compatibility."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, cfg, checkpoint_info = load_fairseq_checkpoint(model_path, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=getattr(args, 'learning_rate', 1e-4))
    
    return model, model.feature_extractor, optimizer, device

def load_original_data2vec_audio_model(model_name="facebook/data2vec-audio-base"):
    model = Data2VecAudioModel.from_pretrained(model_name)
    optimizer = torch.optim.Adam(model.feature_extractor.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), model.feature_extractor,optimizer, device

def normalize_to_audio_range(df):
    df = df.copy()
    df = df.apply(lambda row: [2 * float(v) - 1 for v in row])
    return df

# Collate function for DataLoader
def simple_collate_fn(batch):

    # fixme: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach()
    data = [torch.tensor(sample["data"], dtype=torch.float32) for sample in batch]
    masked_data = [torch.tensor(sample["masked_data"], dtype=torch.float32) for sample in batch]
    return {
        "data": torch.stack(data),
        "masked_data": torch.stack(masked_data)
    }


TARGET_LENGTH = 16000  # 16 kHz


import numpy as np
import scipy.signal
import torch.nn.functional as F

SOURCE_LENGTH = 245
TARGET_LENGTH = 16000  # 1 second at 16kHz

# Preprocessing for pre-trained model

def prepare_resampled_dataloader(df, interpolate_to_16k=True, batch_size=8):
    """
    Takes a pandas DataFrame with N samples of length 245,
    stretches each row using resample_to_16k(), and returns a DataLoader.

    Args:
        df (pd.DataFrame): DataFrame where each row is a sample of length 245.

    Returns:
        DataLoader: Torch-compatible DataLoader with resampled data.
    """
    resampled_tensors = []
    df = normalize_to_audio_range(df["data"])
    for i, row in df.iterrows():
        sample_dict = {"data": row.values.tolist()}
        if interpolate_to_16k:
            resampled = resample_to_16k(sample_dict)["data"]
        else:
            resampled = sample_dict["data"]
        resampled_tensors.append(torch.tensor(resampled, dtype=torch.float32))

    all_data = torch.stack(resampled_tensors)  # shape: [N, 16000]
    dataloader = DataLoader(all_data, batch_size=batch_size, collate_fn=simple_collate_fn)
    return dataloader

def convert_flac_to_wav(input_root, output_root): # used to tests LibSpeech dataset
    """
    Convert all .flac files in the given directory (and subdirectories) to .wav format.
    Saves the .wav files in the same location as the .flac files.

    Args:
        root (str): Root directory to search for .flac files.
    """
    import os, soundfile as sf
    from glob import glob
    import torchaudio
    from tqdm import tqdm

    target_sr = 16000
    # Walk through all subdirectories
    for root, _, files in os.walk(input_root):
        for fname in tqdm(files, desc=f"Converting {root}", leave=False):
            if not fname.endswith(".flac"):
                continue

            in_path = os.path.join(root, fname)

            # Create parallel output folder structure
            rel_dir = os.path.relpath(root, input_root)
            out_dir = os.path.join(output_root, rel_dir)
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".wav")

            # Skip if already converted
            if os.path.exists(out_path):
                continue

            # Load + resample
            wav, sr = torchaudio.load(in_path)
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)

            # Convert to mono (LibriSpeech is stereo)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # Save as WAV
            torchaudio.save(out_path, wav, target_sr)

def create_data2vec_audio_config(): # fixme noy need to review this entire chunk..
    cfg = Data2VecAudioConfig()
    cfg.conv_feature_layers = "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] + [(512, 2, 2)]"
    cfg.encoder_layers = 12
    cfg.average_top_k_layers = 12
    cfg.conv_pos_groups = 16
    cfg.layer_type = "transformer"
    cfg.activation_fn = "gelu"
    cfg.encoder_attention_heads = 12
    cfg.activation_dropout = 0.1

    # === Architecture ===
    cfg.encoder_embed_dim = 768
    cfg.encoder_layers = 12
    cfg.encoder_ffn_embed_dim = 3072
    cfg.encoder_attention_heads = 12
    cfg.activation_fn = "gelu"
    cfg.layer_norm_first = False
    cfg.layerdrop = 0.05

    # === Dropouts ===
    cfg.dropout_input = 0.1
    cfg.dropout_features = 0.1
    cfg.attention_dropout = 0.1
    cfg.activation_dropout = 0.0
    cfg.encoder_layerdrop = 0.05
    cfg.dropout = 0.1

    # === Feature extractor (conv frontend) ===
    cfg.extractor_mode = "layer_norm"
    cfg.conv_feature_layers = "[(512,10,5)] + [(512,3,2)]*4 + [(512,2,2)] + [(512,2,2)]"
    cfg.conv_bias = False
    cfg.feature_grad_mult = 0.1

    # === Positional convolution ===
    cfg.conv_pos = 128
    cfg.conv_pos_groups = 16
    cfg.conv_pos_depth = 5
    cfg.conv_pos_pre_ln = False

    # === Masking parameters ===
    cfg.mask_prob = 0.65
    cfg.mask_selection = "static"
    cfg.mask_other = 0
    cfg.mask_length = 10
    cfg.no_mask_overlap = False
    cfg.mask_min_space = 1
    cfg.mask_channel_prob = 0.0
    cfg.mask_channel_length = 64
    cfg.mask_channel_before = False
    cfg.mask_channel_selection = "static"
    cfg.mask_channel_other = 0
    cfg.no_mask_channel_overlap = False
    cfg.mask_channel_min_space = 1

    # === EMA (teacher / self-distillation) ===
    cfg.ema_decay = 0.999
    cfg.ema_end_decay = 0.9999
    cfg.ema_anneal_end_step = 75000
    cfg.ema_same_dtype = True
    cfg.ema_encoder_only = False

    # === Loss / reconstruction ===
    cfg.loss_beta = 0.0
    cfg.loss_scale = None
    cfg.recon_loss = 0.0
    cfg.recon_dim = 0
    cfg.d2v_loss = 1.0
    cfg.mean_loss = False
    cfg.reconstruct_all = False
    cfg.min_target_var = 0.1
    cfg.min_pred_var = 0.01

    # === Additional Fairseq args ===
    cfg.qk_scale = None
    cfg.cosine_attention = False
    cfg.max_update = 400000
    cfg.seed = 1
    cfg.encoder_layers_to_keep = None
    cfg.layer_norm_target_layer = False
    cfg.batch_norm_target_layer = False
    cfg.instance_norm_target_layer = False
    cfg.log_norms = True
    cfg.shared_decoder = None
    cfg.dropout_input = 0.1
    cfg.dropout_features = 0.1

    # === Data2Vec multi-modality support ===
    cfg.modalities = {"audio": {"feature_encoder_spec": cfg.conv_feature_layers}}
    cfg.supported_modality = "AUDIO"
    cfg.mae_init = False
    cfg.bert_init = True
    cfg.skip_ema = False
    cfg.cls_loss = 0.0
    cfg.alt_cls_targets = False
    cfg.decoder_group = False

    cfg.required_seq_len_multiple = 2  # Fairseq uses this to pad inputs to a multiple of X
    cfg.checkpoint_activations = False  # controls activation checkpointing
    cfg.offload_activations = False  # for memory saving
    cfg.encoder_layerdrop = 0.0  # layer dropout probability
    cfg.feature_grad_mult = 1.0  # gradient scaling for feature extractor
    cfg.extractor_mode = "layer_norm"  # ensures correct mode for ConvFeatureExtractionModel
    cfg.ema_transformer_only = False

    return cfg

def apply_masking(original, mask_ratio=0.15, masking_type="random"):
    """
    Applies masking to a 1D tensor.

    Args:
        original (Tensor): 1D input tensor.
        mask_ratio (float): Ratio of elements to mask.
        masking_type (str): "random", "span", or "low_energy".
        span_length (int): Only used for span masking.

    Returns:
        Tensor: Masked version of original.
    """
    masked = original.clone()

    if masking_type == "random" or masking_type == None:
        indices = torch.randperm(masked.shape[0])[:int(mask_ratio * masked.shape[0])]
        masked[indices] = 0.0

    elif masking_type == "grid":
        step = int(1 / mask_ratio)
        for i in range(0, len(masked), step):
            if i < len(masked):
                masked[i] = 0.0

    elif masking_type == "span start":
        span_length = int(mask_ratio * len(masked))
        masked[0:span_length] = 0.0

    elif masking_type == "span end":
        span_length = int(mask_ratio * len(masked))
        masked[len(masked) - span_length + 1:len(masked)] = 0.0

    elif masking_type == "span": # fixme check this function (draw masked spectograms)
        total_to_mask = int(mask_ratio * len(masked))
        max_span_length = len(masked) // 4  # or just set a cap like 40
        masked_so_far = 0
        used = set()

        while masked_so_far < total_to_mask:
            span_length = random.randint(10, total_to_mask)
            if masked_so_far + span_length > total_to_mask:
                span_length = total_to_mask - masked_so_far
            start = random.randint(0, len(masked) - span_length)

            # avoid overlapping spans
            if any(i in used for i in range(start, start + span_length)):
                continue

            for i in range(start, start + span_length):
                masked[i] = 0.0
                used.add(i)
            masked_so_far += span_length

    elif masking_type == "low_energy":
        energy = original.abs()
        threshold = torch.quantile(energy, mask_ratio)
        masked[energy < threshold] = 0.0

    elif masking_type == "high_energy":
        energy = original.abs()
        threshold = torch.quantile(energy, mask_ratio)
        masked[energy > threshold] = 0.0

    # fixme: add other masking techniques here: multi-channel and labels (seed_id)

    else:
        raise ValueError(f"Unknown masking type: {masking_type}")

    return masked

def safe_get(args, attr, default):
    return default if args is None or not hasattr(args, attr) or getattr(args, attr) is None else getattr(args, attr)


def prepare_masked_dataloader(df, interpolate_to_16k=False, args=None):
    mask_ratio = args.mask_ratio
    batch_size = args.batch_size
    masking_type = args.masking_type

    masked_dataset = []
    df["data"] = normalize_to_audio_range(df["data"])
    for i, row in df.iterrows():
        original = torch.tensor(row["data"], dtype=torch.float32)
        masked = apply_masking(original, mask_ratio=mask_ratio, masking_type=masking_type)
        df.at[i,"masked_data"] = masked.tolist()
        masked_dataset.append({
            "data": original,
            "masked_data": masked
        })

    dataloader = DataLoader(masked_dataset, batch_size=batch_size, collate_fn=simple_collate_fn)
    return dataloader, df

def resample_to_16k(sample, original_sr=SOURCE_LENGTH, target_sr=TARGET_LENGTH): # stretches sample by interpolating a string from 245 to 16k to fit pre-trained model
    def resample_tensor(array):
        tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 245]
        resampled = F.interpolate(tensor, size=target_sr, mode='linear', align_corners=True)
        return resampled.squeeze().tolist()  # [16000]

    sample["data"] = resample_tensor(sample["data"])
    # sample["masked_data"] = resample_tensor(sample["masked_data"]) # fixme uncomment if data returns to maksed version
    return sample

def compute_mask_indices(batch_size, sequence_length, mask_prob=0.05, mask_length=10):
    mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool)
    num_masked_spans = int((sequence_length * mask_prob) // mask_length)

    for b in range(batch_size):
        for _ in range(num_masked_spans):
            start = torch.randint(0, sequence_length - mask_length, (1,)).item()
            mask[b, start:start+mask_length] = True
    return mask


def compute_stack_from_input_spectograms(input_df):
    """
    Splits the input DataFrame into stacks based on index // 10.
    Returns a dict: {stack_idx: df_slice}
    """
    input_df = input_df.copy()
    input_df['stack_idx'] = input_df.index // 10  # assign stack index

    return input_df


#--------------- model evaluation functions ----------------#


def evaluate_embeddings(model, feature_extractor, device, dataset, batch_size=4):
    collator = SelfSupervisedDataCollator(feature_extractor, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, pin_memory=False)

    embeddings = []
    model.eval()
    print("Evaluating model embeddings on masked input...")

    # fixme debug
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            outputs = model(**batch["masked_inputs"])
            emb = outputs.last_hidden_state.last_hidden_state.mean(dim=1)  # shape: (B, T, D)
            embeddings.to(device).append(emb.cpu())
            print(f"\nBatch {i+1} — embeddings shape: {emb.shape}")
            print(f"Mean: {emb.mean().item():.4f}, Std: {emb.std().item():.4f}")
            break  # Show only one batch
    # Stack into one tensor: [N, D]
    embeddings = torch.cat(embeddings, dim=0)

    # Compute + plot cosine similarity matrix
    sim_matrix = compute_cosine_similarity_matrix_from_embeddings(embeddings)


def train_model(df, args):

    model, feature_extractor, optimizer, device = load_custom_data2vec_audio_model(args)
    dataloader, masked_dataset = prepare_masked_dataloader(df, interpolate_to_16k=False, args=args)
    model_string = train_feature_extractor_only(model, optimizer, dataloader, device, args.masking_type, args.arch, args.mask_ratio, args.epoch, args.batch_size, args.loss_function, args.run_id)
    model_path = f"{model_string}_model_after_training.pt"
    return model_path

def train_feature_extractor_only(model, optimizer, dataloader, device, mask_type="random", arch_type="conv1d", mask_ratio=0.15, num_epochs=1, batch_size=8, loss="MSE", run_id=-11):
    """
    Train only the feature extractor layer of the model. Assumes all other layers are already frozen.
    """
    model_string = f"experiment_3-mask_type={mask_type}-mask={mask_ratio}-arch={arch_type}-epoch={num_epochs}_batch={batch_size}_loss_fn={loss}_datalen={len(dataloader.dataset)}"
    model_path = f"/mnt5/noy/code/weights/{model_string}"

    wandb.init(project="SpectralFM", name=model_string) # fixme remove once done migrating to mlflow

    torch.save(model.state_dict(), f"{model_path}_model_before_training.pt")

    model.train()

    # fixme: add here conatrastive and triplet loss once I get pseudo-labels (seeds)
    if loss == "mse":
        loss_fn = nn.MSELoss()
    elif loss == "cosine":
        loss_fn = lambda x, y: 1 - nn.functional.cosine_similarity(x, y, dim=-1).mean()

    with mlflow.start_run(run_name="train_feature_extractor_only"):
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in dataloader:
                # Assume batch is a dict with 'data' and 'masked_data'
                masked_inputs = batch["masked_data"].unsqueeze(1).to(device)
                clean_inputs = batch["data"].unsqueeze(1).to(device)

                optimizer.zero_grad()
                # fixme add gradual learning rate increase + look at theirs training loop (consider theirs training loop)
                student_out = model(masked_inputs).last_hidden_state

                with torch.no_grad():
                    teacher_out = model(clean_inputs).last_hidden_state

                loss = loss_fn(student_out, teacher_out)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                wandb.log({"epoch": epoch, "loss": loss})


            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
            wandb.log({"avg_loss": avg_loss})
            wandb.log({"run_id": run_id})




    # Save model
    torch.save(model.state_dict(), f"{model_path}_model_after_training.pt")
    mlflow.log_artifact(f"{model_path}_model_after_training.pt")

    print(f"Model saved to {model_path}_model_after_training.pt")
    return model_string

# fixme change path to something nicer

def load_trained_model(args):

    model, feature_extractor, optimizer, device = load_custom_data2vec_audio_model(args)
    if args.model_path is not None:
        print(f"Loading model weights from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path), strict=False)
    return model, feature_extractor, optimizer, device

def evaluate_embedding_from_model(args, dataloader):
    """
    Loads a saved Data2VecAudioModel from model_path, runs it on the provided dataset, and computes embeddings and similarities.
    model_path : path to raw weights file
    """

    model, feature_extractor, optimizer, device = load_trained_model(args) # load pre-trained weights if provided

    outputs = []
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # If batch is a dict, get "data" or "masked_data", else assume tensor
            if isinstance(batch, dict):
                # Prefer "masked_data" if present, else "data"
                if "masked_data" in batch:
                    input_tensor = batch["masked_data"]
                else:
                    input_tensor = batch["data"]
            else:
                input_tensor = batch
            input_tensor = input_tensor.unsqueeze(1).to(args.device)
            try:
                out = model(input_values=input_tensor)
                emb = out.last_hidden_state.mean(dim=1)  # [B, D]
                # embeddings = embeddings.to(device)
                # embeddings.append(emb.cpu())
                embeddings.append(emb.to(device))
                outputs.append(out.last_hidden_state.to(device))
            except Exception as e:
                print("Error processing batch:", e)
                if isinstance(batch, dict):
                    print("Batch keys:", batch.keys())
                    print("Batch shapes:", {k: v.shape for k, v in batch.items()})
                else:
                    print("Batch shape:", batch.shape)
    embeddings = torch.cat(embeddings, dim=0)

    return outputs, embeddings

def extract_embeddings(model, dataloader, device):
    """
    Extract embeddings from the model for the given dataloader.
    Used to compare embeddings before and after training.
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            # If batch is a dict, get "data" or "masked_data", else assume tensor
            if isinstance(batch, dict):
                # Prefer "masked_data" if present, else "data"
                if "masked_data" in batch:
                    input_tensor = batch["masked_data"]
                else:
                    input_tensor = batch["data"]
            else:
                input_tensor = batch
            out = model(input_values=input_tensor)
            emb = out.last_hidden_state.mean(dim=1)  # [B, D]
            embeddings.append(emb.cpu())
        embeddings = torch.cat(embeddings, dim=0)
    return embeddings