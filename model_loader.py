from transformers import Data2VecAudioModel, AutoFeatureExtractor
import torch
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


# Load Data2Vec Audio model and feature extractor
def load_data2vec_new_model(model_name="facebook/data2vec-audio-base"):
    # config = AutoConfig.from_pretrained(model_name) fixme debug
    config = Data2VecAudioConfig()
    config.conv_feature_layers = [
        [512, 5, 2],  # kernel=5, stride=2
        [512, 3, 2],
        [512, 3, 2],
        [512, 2, 2]
    ]

    model = Data2VecAudioModel(config) # Initialize model with custom config
    # feature_extractor = Wav2Vec2FeatureExtractor(
    #     feature_size=1,
    #     sampling_rate=16000, # fixme can be changed
    #     padding_value=0.0,
    #     return_attention_mask=True,
    #     do_normalize=True
    # )

    print(model.config.conv_feature_layers)
    print("Model conv layers:")
    for i, layer in enumerate(model.feature_extractor.conv_layers):
        print(f"Layer {i}: {layer}")

    #print("Conv layers:", model.config.conv_feature_layers)

    # fixme debug
    #print("Feature extractor:", feature_extractor)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), device

def load_data2vec_audio_model(model_name="facebook/data2vec-audio-base"):
    model = Data2VecAudioModel.from_pretrained(model_name)
    # fixme add all these to a custom function
    # change to 1 layer feature extractor
    model.feature_extractor = CustomFeatureExtractor()
    model.config.do_stft_input = True
    # freeze all layers apart from feature extractor
    for param in model.parameters():
        param.requires_grad = False
    for param in model.feature_extractor.parameters():
        param.requires_grad = True
    # Define optimizer for trainable params
    optimizer = torch.optim.Adam(model.feature_extractor.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), model.feature_extractor,optimizer, device


def normalize_to_audio_range(df):
    df = df.copy()
    df = df.apply(lambda row: [2 * float(v) - 1 for v in row])
    return df

# Collate function for DataLoader
def simple_collate_fn(batch):

    data = [torch.tensor(sample["data"], dtype=torch.float32) for sample in batch]
    masked_data = [torch.tensor(sample["masked_data"], dtype=torch.float32) for sample in batch]
    return {
        "data": torch.stack(data),
        "masked_data": torch.stack(masked_data)
    }


TARGET_LENGTH = 16000  # 16 kHz

# def pad_to_16k(example):
#     data = np.array(example["masked_data"])
#     if len(data) < TARGET_LENGTH:
#         # Pad with zeros at the end
#         padded = np.pad(data, (0, TARGET_LENGTH - len(data)), mode='constant')
#     else:
#         # Truncate if somehow too long
#         padded = data[:TARGET_LENGTH]
#     return {"masked_data": padded.tolist()}

from datasets import Dataset
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

def prepare_masked_dataloader(df, interpolate_to_16k=True, mask_ratio=0.15, batch_size=8):
    masked_dataset = []
    df = normalize_to_audio_range(df)
    for i, row in df.iterrows():
        sample_dict = {"data": row.values.tolist()}
        if interpolate_to_16k:
            sample_dict = resample_to_16k(sample_dict)
        original = torch.tensor(sample_dict["data"], dtype=torch.float32)
        masked = original.clone()
        indices = torch.randperm(masked.shape[0])[:int(mask_ratio * masked.shape[0])]
        masked[indices] = 0.0
        masked_dataset.append({
            "data": original,
            "masked_data": masked
        })

    dataloader = DataLoader(masked_dataset, batch_size=batch_size, collate_fn=simple_collate_fn)
    return dataloader, masked_dataset

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


def train_self_supervised(model, feature_extractor, device, dataset, output_dir="./pretrained_data2vec", num_epochs=5, batch_size=8, lr=1e-4):

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        save_steps=100,
        logging_steps=10,
        learning_rate=lr,
        remove_unused_columns=False
    )

    collator = SelfSupervisedDataCollator(feature_extractor, device)

    trainer = SelfSupervisedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator)
    #print("Masked device:", dataset["input_values"].device)
    print("Starting self-supervised training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(output_dir)
    feature_extractor.save_pretrained(output_dir)
    print(f"Model and feature extractor saved to {output_dir}")

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
            embeddings.append(emb.cpu())
            print(f"\nBatch {i+1} — embeddings shape: {emb.shape}")
            print(f"Mean: {emb.mean().item():.4f}, Std: {emb.std().item():.4f}")
            break  # Show only one batch
    # Stack into one tensor: [N, D]
    embeddings = torch.cat(embeddings, dim=0)

    # Compute + plot cosine similarity matrix
    sim_matrix = compute_cosine_similarity_matrix_from_embeddings(embeddings)


def train_feature_extractor_only(model, optimizer, dataloader, device, mask_ratio=0.15, num_epochs=1, batch_size=8):
    """
    Train only the feature extractor layer of the model. Assumes all other layers are already frozen.
    """
    wandb.init(project="SpectralFM", name=f"experiment-mask{mask_ratio}-epoch{num_epochs}_batch{batch_size}_datalen{len(dataloader.dataset)}")

    model.train()
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            # Assume batch is a dict with 'data' and 'masked_data'
            masked_inputs = batch["masked_data"].unsqueeze(1).to(device)
            clean_inputs = batch["data"].unsqueeze(1).to(device)

            optimizer.zero_grad()

            student_out = model(masked_inputs).last_hidden_state

            with torch.no_grad():
                teacher_out = model(clean_inputs).last_hidden_state

            loss = loss_fn(student_out, teacher_out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            wandb.log({"epoch": epoch, "loss": loss})
            #with torch.no_grad():
            #    for param_k, param_k in zip(model.named_parameters():
            #        param_k.data = ema_decay * param_k.data + (1 - ema_decay) * param_q.data

        avg_loss = total_loss / len(dataloader)
        wandb.log({"avg_loss": avg_loss})
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), f"experiment-mask{mask_ratio}-epoch{num_epochs}_batch{batch_size}_datalen{len(dataloader.dataset)}_feature_extractor_trained.pt")
    print(f"Model saved to experiment-mask{mask_ratio}-epoch{num_epochs}_batch{batch_size}_datalen{len(dataloader.dataset)}_feature_extractor_trained.pt")

def evaluate_embedding_from_model(model, dataloader, model_path, device, batch_size=8):
    """
    Loads a saved Data2VecAudioModel from model_path, runs it on the provided dataset, and computes embeddings and similarities.
    model_path : path to raw weights file
    """

    model.load_state_dict(torch.load(model_path), strict=False)

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
            input_tensor = input_tensor.unsqueeze(1).to(device)
            try:
                out = model(input_values=input_tensor)
                emb = out.last_hidden_state.mean(dim=1)  # [B, D]
                embeddings.append(emb.cpu())
                outputs.append(out.last_hidden_state.cpu())
            except Exception as e:
                print("Error processing batch:", e)
                if isinstance(batch, dict):
                    print("Batch keys:", batch.keys())
                    print("Batch shapes:", {k: v.shape for k, v in batch.items()})
                else:
                    print("Batch shape:", batch.shape)
    embeddings = torch.cat(embeddings, dim=0)

    # Compute + plot cosine similarity matrix
    sim_matrix = compute_cosine_similarity_matrix_from_embeddings(embeddings)

    return outputs