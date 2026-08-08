#!/usr/bin/env python3
"""
Analyze dataset cosine similarity maps and visualize samples.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import random

# Setup paths
sys.path.insert(0, '/mnt5/noy/SpectralFM/fairseq')
sys.path.insert(0, '/mnt5/noy/SpectralFM/code')

def load_wav_file(wav_path):
    """Load a WAV file (which contains spectrogram data)."""
    try:
        data, sr = sf.read(wav_path, dtype='float32')
        return data
    except Exception as e:
        print(f"Error loading {wav_path}: {e}")
        return None

def load_dataset_from_tsv(tsv_path, wav_dir=None):
    """Load all samples from a TSV manifest file."""
    samples = []
    filenames = []
    
    with open(tsv_path, 'r') as f:
        lines = f.readlines()
        root_path = lines[0].strip()  # First line is the root path
        
        # Use root_path from TSV if wav_dir is not provided or if root_path exists
        if wav_dir is None or os.path.exists(root_path):
            base_wav_dir = root_path
        else:
            base_wav_dir = wav_dir
        
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                filename = parts[0]
                wav_path = os.path.join(base_wav_dir, filename)
                
                if os.path.exists(wav_path):
                    data = load_wav_file(wav_path)
                    if data is not None:
                        samples.append(data)
                        filenames.append(filename)
                else:
                    print(f"Warning: File not found: {wav_path}")
    
    return np.array(samples), filenames

def compute_cosine_similarity_map(samples):
    """Compute pairwise cosine similarity matrix."""
    # Flatten each sample to a 1D vector
    flattened = samples.reshape(len(samples), -1)
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(flattened)
    
    return similarity_matrix

def analyze_dataset(dataset_name, dataset_path, output_dir, num_samples=None):
    """Analyze a dataset and create visualizations."""
    print(f"\n{'='*80}")
    print(f"Analyzing dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    print(f"{'='*80}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    tsv_path = os.path.join(dataset_path, 'train.tsv')
    wav_dir = os.path.join(dataset_path, 'wav')
    
    if not os.path.exists(tsv_path):
        print(f"Error: {tsv_path} not found")
        return None
    
    print(f"Loading samples from {tsv_path}...")
    samples, filenames = load_dataset_from_tsv(tsv_path, wav_dir)
    
    # Randomly sample if requested
    if num_samples is not None and len(samples) > num_samples:
        print(f"Randomly selecting {num_samples} samples from {len(samples)} total...")
        indices = random.sample(range(len(samples)), num_samples)
        samples = samples[indices]
        filenames = [filenames[i] for i in indices]
    
    print(f"Loaded {len(samples)} samples")
    print(f"Sample shape: {samples[0].shape}")
    print(f"Data type: {samples.dtype}")
    print(f"Data range: [{samples.min():.4f}, {samples.max():.4f}]")
    print(f"Data mean: {samples.mean():.4f}, std: {samples.std():.4f}")
    
    # Compute cosine similarity map
    print("\nComputing cosine similarity matrix...")
    similarity_matrix = compute_cosine_similarity_map(samples)
    
    # Compute statistics
    # Exclude diagonal (self-similarity = 1.0)
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    off_diagonal = similarity_matrix[mask]
    
    mean_sim = off_diagonal.mean()
    std_sim = off_diagonal.std()
    min_sim = off_diagonal.min()
    max_sim = off_diagonal.max()
    
    print(f"\nCosine Similarity Statistics:")
    print(f"  Mean: {mean_sim:.4f}")
    print(f"  Std:  {std_sim:.4f}")
    print(f"  Min:  {min_sim:.4f}")
    print(f"  Max:  {max_sim:.4f}")
    
    # Save statistics to file
    stats_file = os.path.join(output_dir, f'{dataset_name}_stats.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Path: {dataset_path}\n")
        f.write(f"Number of samples: {len(samples)}\n")
        f.write(f"Sample shape: {samples[0].shape}\n")
        f.write(f"\nData Statistics:\n")
        f.write(f"  Mean: {samples.mean():.4f}\n")
        f.write(f"  Std:  {samples.std():.4f}\n")
        f.write(f"  Min:  {samples.min():.4f}\n")
        f.write(f"  Max:  {samples.max():.4f}\n")
        f.write(f"\nCosine Similarity Statistics (off-diagonal):\n")
        f.write(f"  Mean: {mean_sim:.4f}\n")
        f.write(f"  Std:  {std_sim:.4f}\n")
        f.write(f"  Min:  {min_sim:.4f}\n")
        f.write(f"  Max:  {max_sim:.4f}\n")
    
    print(f"\nSaved statistics to {stats_file}")
    
    # Plot cosine similarity map
    print("\nCreating cosine similarity heatmap...")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto', vmin=-1, vmax=1)
    ax.set_title(f'Cosine Similarity Map: {dataset_name}\n'
                 f'Mean: {mean_sim:.4f}, Std: {std_sim:.4f}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Sample Index', fontsize=12)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    
    similarity_map_file = os.path.join(output_dir, f'{dataset_name}_cosine_similarity_map.png')
    plt.savefig(similarity_map_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity map to {similarity_map_file}")
    
    # Create plots folder for individual samples
    plots_dir = os.path.join(output_dir, f'{dataset_name}_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot all samples
    print(f"\nPlotting {len(samples)} samples...")
    num_samples_to_plot = len(samples)
    
    # Determine grid size for subplots
    cols = 10
    rows = (num_samples_to_plot + cols - 1) // cols
    
    # Create a large figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 2*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx in range(num_samples_to_plot):
        ax = axes[idx]
        sample = samples[idx]
        
        # Plot the spectrogram (assuming it's a 1D array representing a spectrogram)
        ax.plot(sample)
        ax.set_title(f'Sample {idx+1}\n{os.path.basename(filenames[idx])[:30]}...', 
                    fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_samples_to_plot, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{dataset_name}: All {num_samples_to_plot} Samples', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    all_samples_file = os.path.join(plots_dir, f'{dataset_name}_all_samples.png')
    plt.savefig(all_samples_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved all samples plot to {all_samples_file}")
    
    # Also save individual sample plots
    print(f"Saving individual sample plots...")
    for idx in range(num_samples_to_plot):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(samples[idx])
        ax.set_title(f'Sample {idx+1}: {os.path.basename(filenames[idx])}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Frame', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        individual_file = os.path.join(plots_dir, f'sample_{idx+1:03d}.png')
        plt.savefig(individual_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved individual plots to {plots_dir}")
    
    return {
        'samples': samples,
        'filenames': filenames,
        'similarity_matrix': similarity_matrix,
        'mean_sim': mean_sim,
        'std_sim': std_sim,
        'stats': {
            'mean': samples.mean(),
            'std': samples.std(),
            'min': samples.min(),
            'max': samples.max(),
        }
    }

def main():
    """Main analysis function."""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    base_output_dir = '/mnt5/noy/SpectralFM/scripts/dataset_analysis'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Analyze single_channel_100 dataset
    dataset_100_path = '/mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_100'
    output_100_dir = os.path.join(base_output_dir, 'single_channel_100')
    
    results_100 = analyze_dataset(
        'single_channel_100',
        dataset_100_path,
        output_100_dir
    )
    
    # Analyze 100 random samples from single_channel_all
    dataset_all_path = '/mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_all'
    output_all_dir = os.path.join(base_output_dir, 'single_channel_all_100_random')
    
    results_all = analyze_dataset(
        'single_channel_all_100_random',
        dataset_all_path,
        output_all_dir,
        num_samples=100
    )
    
    # Compare the two datasets
    if results_100 is not None and results_all is not None:
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"Dataset: single_channel_100")
        print(f"  Samples: {len(results_100['samples'])}")
        print(f"  Cosine Similarity - Mean: {results_100['mean_sim']:.4f}, Std: {results_100['std_sim']:.4f}")
        print(f"  Data - Mean: {results_100['stats']['mean']:.4f}, Std: {results_100['stats']['std']:.4f}")
        
        print(f"\nDataset: single_channel_all (100 random samples)")
        print(f"  Samples: {len(results_all['samples'])}")
        print(f"  Cosine Similarity - Mean: {results_all['mean_sim']:.4f}, Std: {results_all['std_sim']:.4f}")
        print(f"  Data - Mean: {results_all['stats']['mean']:.4f}, Std: {results_all['stats']['std']:.4f}")
        
        print(f"\nDifference in Cosine Similarity:")
        print(f"  Mean difference: {results_all['mean_sim'] - results_100['mean_sim']:.4f}")
        print(f"  Std difference: {results_all['std_sim'] - results_100['std_sim']:.4f}")
        
        # Save comparison
        comparison_file = os.path.join(base_output_dir, 'comparison.txt')
        with open(comparison_file, 'w') as f:
            f.write("DATASET COMPARISON\n")
            f.write("="*80 + "\n\n")
            f.write(f"Dataset: single_channel_100\n")
            f.write(f"  Samples: {len(results_100['samples'])}\n")
            f.write(f"  Cosine Similarity - Mean: {results_100['mean_sim']:.4f}, Std: {results_100['std_sim']:.4f}\n")
            f.write(f"  Data - Mean: {results_100['stats']['mean']:.4f}, Std: {results_100['stats']['std']:.4f}\n")
            f.write(f"\nDataset: single_channel_all (100 random samples)\n")
            f.write(f"  Samples: {len(results_all['samples'])}\n")
            f.write(f"  Cosine Similarity - Mean: {results_all['mean_sim']:.4f}, Std: {results_all['std_sim']:.4f}\n")
            f.write(f"  Data - Mean: {results_all['stats']['mean']:.4f}, Std: {results_all['stats']['std']:.4f}\n")
            f.write(f"\nDifference in Cosine Similarity:\n")
            f.write(f"  Mean difference: {results_all['mean_sim'] - results_100['mean_sim']:.4f}\n")
            f.write(f"  Std difference: {results_all['std_sim'] - results_100['std_sim']:.4f}\n")
        
        print(f"\nSaved comparison to {comparison_file}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {base_output_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
