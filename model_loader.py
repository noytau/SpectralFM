from transformers import AutoFeatureExtractor, Data2VecVisionModel, Data2VecAudioForAudioFrameClassification
from datasets import load_dataset
import numpy as np
import random


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

