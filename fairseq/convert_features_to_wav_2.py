import pickle
import os
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
pkl_path = '/mnt5/noy/nova_samples/one_chnl/spectra0000_batch0.pkl'  # Update this to your 1M file
output_wav_dir = '/mnt5/noy/fairseq/data/single_channel_1m/wavs'
manifest_dir = '/mnt5/noy/fairseq/data/single_channel_1m'
validation_fraction = 0.05  # 5% for validation (50k files), 95% for training
# ---------------------

def setup_directories():
    os.makedirs(output_wav_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

def load_and_parse_columns(path):
    print(f"--- Loading Pickle: {path} ---")
    try:
        with open(path, 'rb') as f:
            df = pickle.load(f)
        print(f"Loaded DataFrame with {len(df)} rows.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to open pickle file. {e}")
        exit(1)

    # 2. ROBUST COLUMN PARSING
    # Reuse logic to find columns like '0:0', '0:244'
    raw_cols = [c for c in df.columns if isinstance(c, str) and ':' in c]

    def get_sort_key(col_name):
        try:
            parts = col_name.split(':')
            if len(parts) >= 2:
                return int(parts[1])
            return -1
        except ValueError:
            return -1

    feature_cols = sorted(raw_cols, key=get_sort_key)
    
    if not feature_cols:
        print("CRITICAL: No feature columns found.")
        exit(1)
        
    print(f"Identified {len(feature_cols)} feature columns.")
    return df, feature_cols

def process_split(df_subset, split_name, feature_cols):
    """
    Writes WAV files and generates the TSV manifest for a specific split.
    """
    tsv_path = os.path.join(manifest_dir, f"{split_name}.tsv")
    print(f"\nProcessing split: {split_name} ({len(df_subset)} samples)")
    print(f"Writing manifest to: {tsv_path}")

    # Extract numpy matrix for speed
    matrix = df_subset[feature_cols].values
    # Keep original indices to ensure unique filenames across splits
    original_indices = df_subset.index.tolist()

    with open(tsv_path, "w") as tsv:
        # Fairseq Requirement: First line is the root directory
        tsv.write(output_wav_dir + "\n")
        
        for i, original_idx in tqdm(enumerate(original_indices), total=len(original_indices)):
            try:
                # Convert to float32
                data_row = matrix[i].astype(np.float32)
                
                # Generate unique filename using the original DataFrame index
                fname = f"spec_{original_idx}.wav"
                save_path = os.path.join(output_wav_dir, fname)
                
                # Save as WAV with FLOAT subtype
                # Sample rate 16000 is standard for Data2Vec Audio
                sf.write(save_path, data_row, 16000, subtype='FLOAT')
                
                # Write to Manifest: filename \t n_frames
                tsv.write(f"{fname}\t{len(data_row)}\n")
                
            except Exception as e:
                print(f"Error on row {i} (original_idx {original_idx}): {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    setup_directories()
    
    # 1. Load Data
    df, feature_cols = load_and_parse_columns(pkl_path)

    # 2. Create Splits
    # We shuffle the data to ensure the validation set is representative
    print(f"Splitting data: {100 - validation_fraction*100}% Train, {validation_fraction*100}% Valid")
    train_df, valid_df = train_test_split(df, test_size=validation_fraction, random_state=42, shuffle=True)
    
    # Free up memory by deleting the original large dataframe
    del df 
    
    # 3. Process Validation Set (Do this first to ensure we have a valid set quickly)
    process_split(valid_df, "valid", feature_cols)
    
    # 4. Process Training Set
    process_split(train_df, "train", feature_cols)

    print("\n--- DONE ---")
    print(f"Wav files saved to: {output_wav_dir}")
    print(f"Manifests saved to: {manifest_dir}")
