#!/usr/bin/env python3
"""Create subsets from single_channel_all and copy to RunAI storage."""

import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

# Configuration
SOURCE_DIR = Path("/mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_all")
BASE_TARGET_DIR = Path("/mnt5/noy/SpectralFM/fairseq/data/nova_data")
VALID_PERCENT = 0.01  # 1% validation
SEED = 42
SAMPLE_LENGTH = 245

def create_subset(num_samples, subset_name):
    """Create a subset of samples and manifest files."""
    random.seed(SEED)
    
    target_dir = BASE_TARGET_DIR / subset_name
    target_wav_dir = target_dir / "wav"
    target_wav_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all wav files from source
    source_wav_dir = SOURCE_DIR / "wav"
    if not source_wav_dir.exists():
        source_wav_dir = SOURCE_DIR / "wavs"
    
    if not source_wav_dir.exists():
        raise FileNotFoundError(f"Could not find wav directory in {SOURCE_DIR}")
    
    all_wav_files = list(source_wav_dir.glob("*.wav"))
    print(f"Found {len(all_wav_files)} wav files in source directory")
    
    if len(all_wav_files) < num_samples:
        raise ValueError(f"Not enough files: found {len(all_wav_files)}, need {num_samples}")
    
    # Randomly select files (using same seed ensures no overlap with previous subsets)
    # But we need different samples, so we'll shuffle with a seed based on subset name
    random.seed(SEED + hash(subset_name) % 1000)
    selected_files = random.sample(all_wav_files, num_samples)
    print(f"Selected {len(selected_files)} files for {subset_name}")
    
    # Copy selected files to target directory
    print(f"Copying files to {target_wav_dir}...")
    for wav_file in selected_files:
        shutil.copy2(wav_file, target_wav_dir / wav_file.name)
    print(f"Copied {len(selected_files)} files")
    
    # Split into train and valid (1% validation)
    random.shuffle(selected_files)
    num_valid = max(1, int(len(selected_files) * VALID_PERCENT))
    valid_files = selected_files[:num_valid]
    train_files = selected_files[num_valid:]
    
    print(f"Train: {len(train_files)} samples")
    print(f"Valid: {len(valid_files)} samples")
    
    # Get the absolute path to wav directory for the manifest
    wav_dir_abs = target_wav_dir.resolve()
    
    # Write train.tsv
    train_path = target_dir / "train.tsv"
    with open(train_path, "w") as f:
        f.write(f"{wav_dir_abs}\n")
        for wav_file in train_files:
            f.write(f"{wav_file.name}\t{SAMPLE_LENGTH}\n")
    print(f"Written: {train_path}")
    
    # Write valid.tsv
    valid_path = target_dir / "valid.tsv"
    with open(valid_path, "w") as f:
        f.write(f"{wav_dir_abs}\n")
        for wav_file in valid_files:
            f.write(f"{wav_file.name}\t{SAMPLE_LENGTH}\n")
    print(f"Written: {valid_path}")
    
    return target_dir

def copy_to_runai(local_dir, subset_name):
    """Copy directory to RunAI storage using kubectl cp."""
    print(f"\n{'='*60}")
    print(f"Copying {subset_name} to RunAI storage...")
    print(f"{'='*60}")
    
    # Create tar file
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_tar:
        tar_path = tmp_tar.name
    
    try:
        print(f"Creating tar file: {tar_path}")
        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(local_dir, arcname=subset_name)
        
        # Submit a RunAI job to receive and extract the files
        job_name = f"spectralfm-copy-{subset_name.replace('_', '-')}"
        
        print(f"Submitting RunAI job: {job_name}")
        subprocess.run([
            'runai', 'delete', 'job', job_name, '-p', 'raja'
        ], capture_output=True)
        
        subprocess.run([
            'runai', 'submit', job_name,
            '--image', 'ubuntu:22.04',
            '--gpu', '0',
            '--project', 'raja',
            '--cpu', '1',
            '--memory', '4Gi',
            '--existing-pvc', 'claimname=storage,path=/storage',
            '--interactive',
            '--command', '--',
            'sleep', '3600'
        ], check=True)
        
        # Wait for job to be ready
        print("Waiting for job to be ready...")
        import time
        max_wait = 60
        waited = 0
        while waited < max_wait:
            result = subprocess.run([
                'runai', 'describe', 'job', job_name, '-p', 'raja'
            ], capture_output=True, text=True)
            if 'Running' in result.stdout:
                break
            time.sleep(5)
            waited += 5
        if waited >= max_wait:
            raise RuntimeError(f"Job {job_name} did not start within {max_wait} seconds")
        time.sleep(10)  # Give it a bit more time to fully initialize
        
        # Get pod name (pods are named like job_name-0-0)
        result = subprocess.run([
            'kubectl', 'get', 'pods', '-n', 'runai-raja', '-o', 'jsonpath={range .items[*]}{.metadata.name}{"\\n"}{end}'
        ], capture_output=True, text=True, check=True)
        pod_name = None
        for line in result.stdout.strip().split('\n'):
            if line.startswith(job_name):
                # Check if pod is running
                phase_result = subprocess.run([
                    'kubectl', 'get', 'pod', line, '-n', 'runai-raja', '-o', 'jsonpath={.status.phase}'
                ], capture_output=True, text=True)
                if phase_result.stdout.strip() == 'Running':
                    pod_name = line
                    break
        if not pod_name:
            raise RuntimeError(f"Could not find running pod for job {job_name}")
        
        # Copy tar file to pod
        print(f"Copying tar file to pod {pod_name}...")
        subprocess.run([
            'kubectl', 'cp', tar_path,
            f'runai-raja/{pod_name}:/tmp/{subset_name}.tar.gz'
        ], check=True)
        
        # Extract and copy files
        print("Extracting files in RunAI storage...")
        subprocess.run([
            'runai', 'exec', job_name, '-p', 'raja', '--',
            'bash', '-c', f"""
            cd /storage/noy/SpectralFM/fairseq/data/nova_data &&
            mkdir -p {subset_name} &&
            cd /tmp &&
            tar xzf {subset_name}.tar.gz &&
            cp -r {subset_name}/* /storage/noy/SpectralFM/fairseq/data/nova_data/{subset_name}/ &&
            cd /storage/noy/SpectralFM/fairseq/data/nova_data/{subset_name} &&
            sed -i '1s|.*|/storage/noy/SpectralFM/fairseq/data/nova_data/{subset_name}/wav|' train.tsv &&
            sed -i '1s|.*|/storage/noy/SpectralFM/fairseq/data/nova_data/{subset_name}/wav|' valid.tsv &&
            echo 'Copy completed for {subset_name}' &&
            ls -lh /storage/noy/SpectralFM/fairseq/data/nova_data/{subset_name}/ &&
            find /storage/noy/SpectralFM/fairseq/data/nova_data/{subset_name}/wav -type f | wc -l
            """
        ], check=True)
        
        # Clean up job
        print(f"Cleaning up job {job_name}...")
        subprocess.run([
            'runai', 'delete', 'job', job_name, '-p', 'raja'
        ], capture_output=True)
        
        print(f"✓ Successfully copied {subset_name} to RunAI storage")
        
    finally:
        # Clean up tar file
        if os.path.exists(tar_path):
            os.remove(tar_path)

def main():
    subsets = [
        (1000, "single_channel_1k"),
        (10000, "single_channel_10k"),
    ]
    
    for num_samples, subset_name in subsets:
        print(f"\n{'='*60}")
        print(f"Processing {subset_name} ({num_samples} samples)")
        print(f"{'='*60}")
        
        # Create subset locally
        local_dir = create_subset(num_samples, subset_name)
        
        # Copy to RunAI storage
        copy_to_runai(local_dir, subset_name)
    
    print(f"\n{'='*60}")
    print("All subsets created and copied successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
