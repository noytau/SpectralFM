#!/usr/bin/env python3
"""
Select 100 samples from single_channel_all with highest standard deviation.
Create manifest files and copy samples to new directory.
"""

import os
import sys
import numpy as np
import soundfile as sf
import shutil
from pathlib import Path

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

def load_all_samples_from_tsv(tsv_path, top_n=100):
    """Load all samples from a TSV manifest file and keep top N by std (memory efficient)."""
    # Keep only top N samples in memory
    top_samples = []  # List of (filename, std, data) tuples
    min_std_in_top = -1.0
    
    with open(tsv_path, 'r') as f:
        lines = f.readlines()
        root_path = lines[0].strip()  # First line is the root path
        
        print(f"Loading samples from {tsv_path}...")
        print(f"Root path: {root_path}")
        print(f"Keeping top {top_n} samples by std (memory efficient mode)")
        
        total_lines = len(lines) - 1
        processed_count = 0
        loaded_count = 0
        
        for idx, line in enumerate(lines[1:], 1):
            if idx % 100000 == 0:
                print(f"  Processed {idx}/{total_lines} lines, loaded {loaded_count} samples, current min std in top: {min_std_in_top:.4f}...")
            
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                filename = parts[0]
                wav_path = os.path.join(root_path, filename)
                
                if os.path.exists(wav_path):
                    data = load_wav_file(wav_path)
                    if data is not None:
                        std = np.std(data)
                        processed_count += 1
                        
                        # Only keep if it's in top N or better than current minimum
                        if len(top_samples) < top_n:
                            top_samples.append((filename, std, data))
                            top_samples.sort(key=lambda x: x[1], reverse=True)
                            min_std_in_top = top_samples[-1][1] if len(top_samples) == top_n else -1.0
                            loaded_count += 1
                        elif std > min_std_in_top:
                            # Replace the minimum with this one
                            top_samples[-1] = (filename, std, data)
                            top_samples.sort(key=lambda x: x[1], reverse=True)
                            min_std_in_top = top_samples[-1][1]
                            loaded_count += 1
                else:
                    if idx <= 10:  # Only print first few warnings
                        print(f"Warning: File not found: {wav_path}")
    
    print(f"\nProcessed {processed_count} samples, kept top {len(top_samples)} by std")
    
    # Extract results
    filenames = [item[0] for item in top_samples]
    stds = np.array([item[1] for item in top_samples])
    samples = np.array([item[2] for item in top_samples])
    
    return samples, filenames, stds, root_path

def select_top_n_by_std(filenames, stds, n=100):
    """Select top N samples by standard deviation."""
    # Get indices sorted by std (descending)
    sorted_indices = np.argsort(stds)[::-1]
    
    # Select top N
    top_indices = sorted_indices[:n]
    
    selected_filenames = [filenames[i] for i in top_indices]
    selected_stds = stds[top_indices]
    
    return selected_filenames, selected_stds, top_indices

def create_manifest(output_dir, root_path, filenames, split_name='train'):
    """Create a TSV manifest file."""
    manifest_path = os.path.join(output_dir, f'{split_name}.tsv')
    
    with open(manifest_path, 'w') as f:
        # Write root path
        f.write(f"{root_path}\n")
        
        # Write file entries
        for filename in filenames:
            # Get file length (number of frames)
            wav_path = os.path.join(root_path, filename)
            if os.path.exists(wav_path):
                data, _ = sf.read(wav_path, dtype='float32')
                length = len(data)
                f.write(f"{filename}\t{length}\n")
            else:
                print(f"Warning: Cannot get length for {wav_path}")
                f.write(f"{filename}\t0\n")
    
    print(f"Created manifest: {manifest_path}")
    return manifest_path

def copy_wav_files(source_root, dest_dir, filenames):
    """Copy WAV files to destination directory."""
    dest_wav_dir = os.path.join(dest_dir, 'wav')
    os.makedirs(dest_wav_dir, exist_ok=True)
    
    print(f"\nCopying {len(filenames)} WAV files to {dest_wav_dir}...")
    
    for idx, filename in enumerate(filenames, 1):
        if idx % 10 == 0:
            print(f"  Copied {idx}/{len(filenames)} files...")
        
        source_path = os.path.join(source_root, filename)
        dest_path = os.path.join(dest_wav_dir, filename)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
        else:
            print(f"Warning: Source file not found: {source_path}")
    
    print(f"Copying complete!")

def main():
    """Main function."""
    # Paths
    source_tsv = '/mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_all/train.tsv'
    output_dir = '/mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_100_var'
    num_samples = 100
    
    print("="*80)
    print("CREATING HIGH VARIANCE SUBSET")
    print("="*80)
    print(f"Source: {source_tsv}")
    print(f"Output: {output_dir}")
    print(f"Number of samples: {num_samples}")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load top N samples by std (already sorted)
    samples, filenames, stds, root_path = load_all_samples_from_tsv(source_tsv, top_n=num_samples)
    
    if len(samples) < num_samples:
        print(f"Warning: Only {len(samples)} samples available, but {num_samples} requested")
        num_samples = len(samples)
    
    # Samples are already sorted by std (descending)
    selected_filenames = filenames
    selected_stds = stds
    
    # Print statistics
    print(f"\nSelected samples statistics:")
    print(f"  Mean std: {selected_stds.mean():.4f}")
    print(f"  Min std:  {selected_stds.min():.4f}")
    print(f"  Max std:  {selected_stds.max():.4f}")
    print(f"  Overall dataset mean std: {stds.mean():.4f}")
    print(f"  Overall dataset max std:  {stds.max():.4f}")
    
    # Save statistics
    stats_file = os.path.join(output_dir, 'selection_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("High Variance Subset Selection Statistics\n")
        f.write("="*80 + "\n\n")
        f.write(f"Source: {source_tsv}\n")
        f.write(f"Total samples in source: {len(samples)}\n")
        f.write(f"Selected samples: {num_samples}\n\n")
        f.write(f"Selected samples std statistics:\n")
        f.write(f"  Mean: {selected_stds.mean():.4f}\n")
        f.write(f"  Std:  {selected_stds.std():.4f}\n")
        f.write(f"  Min:  {selected_stds.min():.4f}\n")
        f.write(f"  Max:  {selected_stds.max():.4f}\n\n")
        f.write(f"Overall dataset std statistics:\n")
        f.write(f"  Mean: {stds.mean():.4f}\n")
        f.write(f"  Std:  {stds.std():.4f}\n")
        f.write(f"  Min:  {stds.min():.4f}\n")
        f.write(f"  Max:  {stds.max():.4f}\n\n")
        f.write(f"Top 10 selected samples:\n")
        for i, (filename, std_val) in enumerate(zip(selected_filenames[:10], selected_stds[:10]), 1):
            f.write(f"  {i}. {filename}: std={std_val:.4f}\n")
    
    print(f"\nSaved statistics to {stats_file}")
    
    # Copy WAV files
    copy_wav_files(root_path, output_dir, selected_filenames)
    
    # Update root path for the new directory
    new_root_path = os.path.join(output_dir, 'wav')
    
    # Create manifest files
    print(f"\nCreating manifest files...")
    create_manifest(output_dir, new_root_path, selected_filenames, 'train')
    
    # Create a small validation set (10 samples)
    if len(selected_filenames) >= 10:
        valid_filenames = selected_filenames[:10]
        create_manifest(output_dir, new_root_path, valid_filenames, 'valid')
    
    print("\n" + "="*80)
    print("SUBSET CREATION COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Selected {num_samples} samples with highest standard deviation")
    print("="*80)

if __name__ == "__main__":
    main()
