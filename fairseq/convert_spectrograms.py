import pickle
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
pkl_path = '/mnt5/noy/nova_samples/one_chnl/spectra0000_batch0.pkl'
tsv_path = '/mnt5/noy/fairseq/data/single_channel_1m/train.tsv'
# ---------------------

output_dir = os.path.dirname(tsv_path)
os.makedirs(output_dir, exist_ok=True)

print(f"Loading pickle from: {pkl_path}")
try:
    df = pickle.load(open(pkl_path, 'rb'))
except Exception as e:
    print(f"CRITICAL ERROR: Could not load pickle. {e}")
    exit(1)

print(f"Loaded DataFrame with {len(df)} rows.")

# 1. IDENTIFY FEATURE COLUMNS
# We filter for columns that look like '0:123' and exclude 'x', 'y'
feature_cols = [c for c in df.columns if ':' in str(c) and c not in ['x', 'y']]

# 2. SORT COLUMNS NUMERICALLY (CRITICAL)
# '0:10' must come after '0:9', not before '0:2'.
def get_col_index(col_name):
    try:
        # Split '0:244' -> takes '244' -> converts to int
        return int(col_name.split(':')[1])
    except:
        return -1

# Sort the columns based on the suffix index
feature_cols = sorted(feature_cols, key=get_col_index)

print(f"Identified {len(feature_cols)} feature columns.")
print(f"First 5 cols: {feature_cols[:5]}")
print(f"Last 5 cols:  {feature_cols[-5:]}")

# Check if we have the expected length
if len(feature_cols) != 245:
    print(f"WARNING: Expected 245 columns, found {len(feature_cols)}.")

# 3. PROCESS AND SAVE
print(f"Extracting and saving to {output_dir}...")

with open(tsv_path, "w") as tsv:
    tsv.write(output_dir + "\n")
    
    # We extract the numpy matrix for all feature columns at once for speed
    # Shape: (Num_Rows, 245)
    all_data = df[feature_cols].values
    
    valid_count = 0
    
    for i in tqdm(range(len(all_data))):
        try:
            # Get the row as float32
            row_array = all_data[i].astype(np.float32)
            
            # Convert to Tensor
            tensor = torch.tensor(row_array)
            
            # Save
            fname = f"spec_{i}.pt"
            torch.save(tensor, os.path.join(output_dir, fname))
            
            # Write to manifest
            tsv.write(f"{fname}\t{tensor.shape[0]}\n")
            valid_count += 1
            
        except Exception as e:
            print(f"Failed on row {i}: {e}")

print(f"--- DONE ---")
print(f"Successfully generated {valid_count} samples.")
print(f"Manifest saved to: {tsv_path}")
