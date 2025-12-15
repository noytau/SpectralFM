import pickle
import torch
import os
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# --- CONFIGURATION ---
pkl_path = '/mnt5/noy/nova_samples/one_chnl/spectra0000_batch0.pkl'
tsv_path = '/mnt5/noy/fairseq/data/single_channel_1m/train.tsv'
# ---------------------

output_dir = os.path.dirname(tsv_path)
os.makedirs(output_dir, exist_ok=True)

print(f"--- Processing {pkl_path} ---")

# 1. LOAD PICKLE ROBUSTLY
try:
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    print(f"Successfully loaded DataFrame with {len(df)} rows.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to open pickle file.")
    print(f"Error details: {e}")
    exit(1)

# 2. ROBUST COLUMN PARSING
# We look for columns that have a colon ':' (e.g., '0:0', '0:244')
# We must split by ':' and use the SECOND number as the sort index.
raw_cols = [c for c in df.columns if isinstance(c, str) and ':' in c]

def get_sort_key(col_name):
    # Splits "0:15" -> ["0", "15"] -> returns integer 15
    try:
        parts = col_name.split(':')
        if len(parts) >= 2:
            return int(parts[1])
        return -1
    except ValueError:
        return -1

# Sort columns numerically (0, 1, 2, ..., 10, 11, ...)
feature_cols = sorted(raw_cols, key=get_sort_key)

# VALIDATION: Check if we found the columns
if len(feature_cols) == 0:
    print("CRITICAL: No columns with ':' found!")
    print(f"First 10 columns in df: {list(df.columns)[:10]}")
    exit(1)

print(f"Found {len(feature_cols)} feature columns.")
print(f"First 3 columns: {feature_cols[:3]} (Should be low numbers)")
print(f"Last 3 columns:  {feature_cols[-3:]} (Should be high numbers)")

# 3. CONVERSION LOOP
print(f"Converting to Float WAVs in {output_dir}...")

with open(tsv_path, "w") as tsv:
    tsv.write(output_dir + "\n")
    
    # Extract values as a matrix (Rows x Features)
    # This is much faster than iterating row-by-row in Pandas
    matrix = df[feature_cols].values
    
    valid_count = 0
    for i in tqdm(range(len(matrix))):
        try:
            # Get the single spectrogram row
            # Shape: (245,)
            data_row = matrix[i].astype(np.float32)
            
            # Save as WAV
            # subtype='FLOAT' is crucial to keep values like -5.4, 10.2
            fname = f"spec_{i}.wav"
            save_path = os.path.join(output_dir, fname)
            sf.write(save_path, data_row, 16000, subtype='FLOAT')
            
            # Write Manifest: filename \t number_of_frames
            tsv.write(f"{fname}\t{len(data_row)}\n")
            valid_count += 1
            
        except Exception as e:
            print(f"Error on row {i}: {e}")

print(f"--- DONE ---")
print(f"Generated {valid_count} files.")
