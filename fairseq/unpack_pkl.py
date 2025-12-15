import pickle
import torch
import os
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
pkl_path = '/mnt5/noy/nova_samples/one_chnl/spectra0000_batch0.pkl'
tsv_path = '/mnt5/noy/fairseq/data/single_channel_1m/train.tsv'

# The directory where we will save the .pt files (derived from TSV path)
output_dir = os.path.dirname(tsv_path)
os.makedirs(output_dir, exist_ok=True)

print(f"--- processing ---")
print(f"Input: {pkl_path}")
print(f"Output Dir: {output_dir}")
print(f"Target TSV: {tsv_path}")

# 1. LOAD DATA
print(f"Loading pickle...")
data = pickle.load(open(pkl_path, 'rb'))

# 2. DETERMINE ITERATOR (Handle Dict vs List)
iterator = None
if isinstance(data, dict):
    print(f"Detected Dictionary (len={len(data)}). Extracting values...")
    iterator = data.values() 
elif isinstance(data, list):
    print(f"Detected List (len={len(data)}).")
    iterator = data
else:
    raise ValueError(f"Unknown data type in pickle: {type(data)}")

# 3. PROCESS AND SAVE
print("Starting conversion...")

with open(tsv_path, "w") as tsv:
    # Fairseq TSV Header: Absolute path to the directory containing files
    tsv.write(output_dir + "\n")
    
    count = 0
    skipped = 0
    
    for i, spec in enumerate(iterator):
        
        # Safety Check: If the pickle contains filenames (strings) instead of arrays
        if isinstance(spec, str):
            if count < 3: print(f"Warning: Item {i} is a string '{spec}'. Skipping.")
            skipped += 1
            continue
            
        # Convert to Torch Tensor
        # (Handles numpy arrays or python lists automatically)
        if not torch.is_tensor(spec):
            try:
                spec = torch.tensor(np.array(spec))
            except Exception as e:
                print(f"Error converting item {i} to tensor: {e}")
                skipped += 1
                continue

        # Save individual tensor file
        fname = f"spec_{i}.pt"
        save_path = os.path.join(output_dir, fname)
        torch.save(spec, save_path)
        
        # Write to Manifest: filename <tab> length
        # Note: spec.shape[0] assumes the first dim is the time/length dimension
        tsv.write(f"{fname}\t{spec.shape[0]}\n")
        
        count += 1
        if count % 10000 == 0:
            print(f"Saved {count} files...")

print("--- DONE ---")
print(f"Successfully saved {count} .pt files.")
if skipped > 0:
    print(f"Skipped {skipped} items (likely strings/invalid).")
print(f"Manifest saved to: {tsv_path}")
