#!/usr/bin/env python3
"""
Prepare LibriSpeech test-clean dataset for fairseq evaluation.

Downloads test-clean.tar.gz from openslr.org/12 and creates a fairseq-compatible
manifest file for evaluation.
"""

import os
import sys
import argparse
import tarfile
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm
import shutil

# Add fairseq to path
_FAIRSEQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fairseq")
if _FAIRSEQ_PATH not in sys.path:
    sys.path.insert(0, _FAIRSEQ_PATH)

def download_file(url, output_path):
    """Download a file with progress bar."""
    import urllib.request
    
    if output_path.exists():
        print(f"[+] File already exists: {output_path}")
        return output_path
    
    print(f"[+] Downloading {url}...")
    urllib.request.urlretrieve(url, output_path, reporthook=lambda blocknum, blocksize, totalsize: None)
    print(f"[+] Downloaded to: {output_path}")
    return output_path

def extract_tar(tar_path, extract_to):
    """Extract tar.gz file."""
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    
    if (extract_to / "LibriSpeech" / "test-clean").exists():
        print(f"[+] Already extracted to: {extract_to}")
        return extract_to / "LibriSpeech" / "test-clean"
    
    print(f"[+] Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_to)
    
    test_clean_path = extract_to / "LibriSpeech" / "test-clean"
    if test_clean_path.exists():
        print(f"[+] Extracted to: {test_clean_path}")
        return test_clean_path
    else:
        raise FileNotFoundError(f"test-clean directory not found after extraction")

def create_fairseq_manifest(test_clean_dir, output_manifest):
    """
    Create a fairseq-compatible manifest file for test-clean.
    
    Format for FileAudioDataset:
    - First line: root directory path (where audio files are relative to)
    - Subsequent lines: id <TAB> n_frames (2 items only)
    - Audio files are found at: root_dir/id.flac (or root_dir/relative_path_from_manifest)
    """
    test_clean_dir = Path(test_clean_dir)
    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    
    # Get the directory containing the manifest (this will be the root for relative paths)
    manifest_dir = output_manifest.parent
    
    print(f"[+] Creating manifest from: {test_clean_dir}")
    print(f"[+] Output manifest: {output_manifest}")
    print(f"[+] Manifest directory (root for relative paths): {manifest_dir}")
    
    # Find all .flac files
    flac_files = sorted(list(test_clean_dir.rglob("*.flac")))
    print(f"[+] Found {len(flac_files)} audio files")
    
    # First line: root directory (absolute path to audio files)
    # FileAudioDataset expects the root_dir to be where audio files are located
    # We'll use absolute path to avoid path resolution issues
    root_dir = str(test_clean_dir.absolute())
    
    manifest_lines = [f"{root_dir}\n"]  # First line is root directory
    
    for flac_path in tqdm(flac_files, desc="Processing audio files"):
        # Get relative path from test_clean_dir
        rel_path = flac_path.relative_to(test_clean_dir)
        
        # Extract sample ID from path: e.g., "61/70968/61-70968-0000.flac" -> "61-70968-0000"
        # LibriSpeech format: speaker-chapter-utterance
        sample_id = rel_path.stem
        
        # Load audio to get frame count
        try:
            waveform, sample_rate = torchaudio.load(str(flac_path))
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            n_frames = waveform.shape[1]
            
            # FileAudioDataset format: id <TAB> n_frames (2 items only)
            # The audio file path is constructed as: root_dir/rel_path
            # But we need to store the relative path from root_dir, not just the ID
            # Actually, FileAudioDataset uses the ID as the filename, so we need the full relative path
            audio_rel_path = str(rel_path).replace('\\', '/')  # Normalize path separators
            
            # Write: audio_relative_path <TAB> n_frames
            manifest_lines.append(f"{audio_rel_path}\t{n_frames}\n")
        except Exception as e:
            print(f"[!] Error processing {flac_path}: {e}")
            continue
    
    # Write manifest file
    with open(output_manifest, 'w') as f:
        f.writelines(manifest_lines)
    
    print(f"[+] Created manifest with {len(manifest_lines)-1} entries (first line is root dir)")
    return output_manifest

def main():
    parser = argparse.ArgumentParser(description="Prepare LibriSpeech test-clean for fairseq evaluation")
    parser.add_argument("--download-dir", type=str, default="/mnt5/noy/SpectralFM/data/librispeech",
                       help="Directory to download and extract test-clean")
    parser.add_argument("--output-dir", type=str, default="/mnt5/noy/SpectralFM/fairseq/data/librispeech_test_clean",
                       help="Directory for fairseq-compatible dataset")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download if tar.gz already exists")
    parser.add_argument("--skip-extract", action="store_true",
                       help="Skip extraction if already extracted")
    
    args = parser.parse_args()
    
    download_dir = Path(args.download_dir)
    output_dir = Path(args.output_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download test-clean.tar.gz
    tar_path = download_dir / "test-clean.tar.gz"
    if not args.skip_download:
        url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
        download_file(url, tar_path)
    else:
        if not tar_path.exists():
            print(f"[!] Error: {tar_path} does not exist. Remove --skip-download to download.")
            return 1
    
    # Extract
    if not args.skip_extract:
        test_clean_dir = extract_tar(tar_path, download_dir)
    else:
        test_clean_dir = download_dir / "LibriSpeech" / "test-clean"
        if not test_clean_dir.exists():
            print(f"[!] Error: {test_clean_dir} does not exist. Remove --skip-extract to extract.")
            return 1
    
    # For fairseq, we can either:
    # 1. Keep audio files in place and use absolute/relative paths in manifest
    # 2. Copy audio files to output directory
    # We'll keep them in place and use relative paths from the manifest directory
    
    # Create manifest file (fairseq expects valid.tsv for validation)
    # We'll create both test-clean.tsv and valid.tsv (same content)
    manifest_path_valid = output_dir / "valid.tsv"
    manifest_path_test = output_dir / "test-clean.tsv"
    
    # Create manifest with audio files in their original location
    # The manifest will be in output_dir, so we need relative paths from there
    create_fairseq_manifest(test_clean_dir, manifest_path_valid)
    
    # Also create test-clean.tsv as a copy (for reference)
    shutil.copy2(manifest_path_valid, manifest_path_test)
    
    print(f"[+] Created both valid.tsv and test-clean.tsv")
    
    print(f"\n[+] Dataset prepared successfully!")
    print(f"[+] Dataset directory: {output_dir}")
    print(f"[+] Manifest files: {manifest_path_valid}, {manifest_path_test}")
    print(f"[+] Audio files location: {test_clean_dir}")
    print(f"\n[+] Use this path for evaluation:")
    print(f"    {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
