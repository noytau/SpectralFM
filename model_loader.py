from transformers import Data2VecAudioModel, AutoFeatureExtractor
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor, Data2VecAudioModel, AutoConfig, TrainingArguments, Data2VecAudioConfig
import torch
from datasets import Dataset
import numpy as np
import random
from trainer import SelfSupervisedDataCollator, SelfSupervisedTrainer

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

def plot_masked_dataset_statistics(dataset, output_dir="plots"):
    """
    Plots statistics and visualizations from a masked Hugging Face dataset.

    Args:
        dataset: A Hugging Face dataset with 'data', 'masked_data', and 'mask_indices'.
        output_dir (str): Directory where the plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    data = np.array(dataset["data"])
    masked_data = np.array(dataset["masked_data"])
    mask_counts = np.zeros(data.shape[1], dtype=int)

    for _, col in dataset["mask_indices"]:
        mask_counts[col] += 1

    # 1. Plot mean and std of original data
    plt.figure()
    plt.plot(np.mean(data, axis=0), label='Mean')
    plt.fill_between(range(data.shape[1]),
                     np.mean(data, axis=0) - np.std(data, axis=0),
                     np.mean(data, axis=0) + np.std(data, axis=0),
                     alpha=0.3, label='Std')
    plt.title("Mean and Std of Original Data")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "mean_std_data.png"))
    plt.close()

    # 2. Plot mean and std of masked data
    plt.figure()
    plt.plot(np.mean(masked_data, axis=0), label='Mean')
    plt.fill_between(range(masked_data.shape[1]),
                     np.mean(masked_data, axis=0) - np.std(masked_data, axis=0),
                     np.mean(masked_data, axis=0) + np.std(masked_data, axis=0),
                     alpha=0.3, label='Std')
    plt.title("Mean and Std of Masked Data")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "mean_std_masked_data.png"))
    plt.close()

    # 3. Plot mean and std of difference
    diff = data - masked_data
    plt.figure()
    plt.plot(np.mean(diff, axis=0), label='Mean Difference')
    plt.fill_between(range(data.shape[1]),
                     np.mean(diff, axis=0) - np.std(diff, axis=0),
                     np.mean(diff, axis=0) + np.std(diff, axis=0),
                     alpha=0.3, label='Std Difference')
    plt.title("Mean and Std of Difference (Original - Masked)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "mean_std_difference.png"))
    plt.close()

    # 4. Plot first 5 original vs masked spectrograms
    for i in range(min(5, data.shape[0])):
        plt.figure()
        plt.plot(data[i], label='Original')
        plt.plot(masked_data[i], label='Masked', linestyle='--')
        plt.title(f"Spectrogram {i}: Original vs Masked")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"spectrogram_{i}_compare.png"))
        plt.close()

    # 5. Plot number of times each index was masked
    plt.figure()
    plt.bar(range(len(mask_counts)), mask_counts)
    plt.title("Mask Frequency per Index (0-244)")
    plt.xlabel("Index")
    plt.ylabel("Masked Count")
    plt.savefig(os.path.join(output_dir, "mask_frequency.png"))
    plt.close()

    # 6. Plot scatter plot of std vs mean for original and masked data and the difference
    plt.figure()
    plt.scatter(np.mean(data, axis=1), np.std(data, axis=1), s=10, alpha=0.7, color='blue', label='Original')
    plt.scatter(np.mean(masked_data, axis=1), np.std(masked_data, axis=1), s=10, alpha=0.7, color='orange', label='Masked')
    plt.xlabel("Mean")
    plt.ylabel("Standard Deviation")
    plt.title("Std vs Mean (Original vs Masked)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "scatter_std_vs_mean_comparison.png"))
    plt.close()


# Load Data2Vec Audio model and feature extractor
def load_data2vec_audio_model(model_name="facebook/data2vec-audio-base"):
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

def normalize_to_audio_range(x):
    return [2 * v - 1 for v in x]  # maps [0,1] → [-1,1]

# Collate function for DataLoader
def simple_collate_fn(batch, feature_extractor, device):
    inputs = [sample["masked_data"] for sample in batch]
    processed = feature_extractor(inputs, sampling_rate=16000, return_tensors="pt")
    return {k: v.to(device) for k, v in processed.items()}

# Run model on masked dataset
def run_model_on_masked_dataset(dataset, model, feature_extractor, device, batch_size=8):
    dataset = Dataset.from_dict(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda b: simple_collate_fn(b, feature_extractor, device))
    print(f"type of dataloader: {type(dataloader)}, type of dataset: {type(dataset)}")
    print(dataset[0])
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            try:
                out = model(**batch)
                outputs.append(out.last_hidden_state.cpu())
            except Exception as e:
                print("Error processing batch:", e)
                print("Batch keys:", batch.keys())
                print("Batch shapes:", {k: v.shape for k, v in batch.items()})
    return outputs

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

    model.eval()
    print("Evaluating model embeddings on masked input...")

    # fixme debug
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            outputs = model(**batch["masked_inputs"])
            embeddings = outputs.last_hidden_state  # shape: (B, T, D)
            print(f"\nBatch {i+1} — embeddings shape: {embeddings.shape}")
            print(f"Mean: {embeddings.mean().item():.4f}, Std: {embeddings.std().item():.4f}")
            break  # Show only one batch

# Example usage:
# model, feature_extractor, device = load_data2vec_audio_model()
# features = run_model_on_masked_dataset(masked_dataset, model, feature_extractor, device)

