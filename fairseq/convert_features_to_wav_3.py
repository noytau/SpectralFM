import os
import glob
import pickle
import numpy as np
import soundfile as sf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
input_pkl_dir = '/mnt5/noy/nova_samples/full_chnl/'   # Directory containing multiple .pkl files
output_wav_dir = '/mnt5/noy/fairseq/data/single_channel_all/wavs'
manifest_dir = '/mnt5/noy/fairseq/data/single_channel_all'
validation_fraction = 0.05  # 5% validation
# ---------------------

def setup_directories():
    os.makedirs(output_wav_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

def get_feature_columns(df):
    """
    Parses column names to find features like '0:0', '0:1' etc.
    Returns the sorted list of feature column names.
    """
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
    return feature_cols

def process_pickle_file(pkl_path):
    """
    Loads a single pickle, converts rows to WAVs, and returns a list of metadata.
    Metadata format: [(filename, length), (filename, length), ...]
    """
    pkl_name = os.path.splitext(os.path.basename(pkl_path))[0]
    file_metadata = []

    print(f"--- Processing: {pkl_name} ---")
    
    try:
        with open(pkl_path, 'rb') as f:
            df = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading {pkl_name}: {e}")
        return []

    feature_cols = get_feature_columns(df)
    if not feature_cols:
        print(f"⚠️ No feature columns found in {pkl_name}. Skipping.")
        return []

    # Extract numpy matrix for speed
    matrix = df[feature_cols].values
    original_indices = df.index.tolist()

    # Iterate and save WAVs
    for i, original_idx in enumerate(original_indices):
        try:
            # Create a unique filename: {pkl_name}_{original_index}.wav
            # Example: spectra_batch0_spec_1024.wav
            fname = f"{pkl_name}_spec_{original_idx}.wav"
            save_path = os.path.join(output_wav_dir, fname)

            # Check if file exists to save write time (optional, safer to overwrite)
            data_row = matrix[i].astype(np.float32)
            
            # Save as WAV (FLOAT format for precision)
            sf.write(save_path, data_row, 16000, subtype='FLOAT')
            
            # Add to metadata list
            file_metadata.append((fname, len(data_row)))
            
        except Exception as e:
            print(f"Error converting row {i} in {pkl_name}: {e}")

    # Free memory immediately
    del df
    del matrix
    
    return file_metadata

def write_manifest(metadata_list, split_name):
    """
    Writes the list of (filename, length) to a Fairseq-compliant TSV.
    """
    tsv_path = os.path.join(manifest_dir, f"{split_name}.tsv")
    print(f"Writing {split_name} manifest ({len(metadata_list)} samples) to: {tsv_path}")
    
    with open(tsv_path, "w") as tsv:
        # Header: Root directory
        tsv.write(output_wav_dir + "\n")
        
        for fname, length in metadata_list:
            tsv.write(f"{fname}\t{length}\n")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    setup_directories()
    
    # 1. Gather all .pkl files
    pkl_files = glob.glob(os.path.join(input_pkl_dir, "*.pkl"))
    if not pkl_files:
        print(f"CRITICAL: No .pkl files found in {input_pkl_dir}")
        exit(1)
        
    print(f"Found {len(pkl_files)} pickle files. Starting conversion...")

    # 2. Process all files and collect metadata
    all_samples_metadata = []
    
    for pkl_file in tqdm(pkl_files, desc="Processing Pickle Files"):
        metadata = process_pickle_file(pkl_file)
        all_samples_metadata.extend(metadata)

    print(f"\nTotal extracted samples: {len(all_samples_metadata)}")

    # 3. Create Train/Valid Splits
    if len(all_samples_metadata) > 0:
        print(f"Splitting data: {100 - validation_fraction*100}% Train, {validation_fraction*100}% Valid")
        
        train_meta, valid_meta = train_test_split(
            all_samples_metadata, 
            test_size=validation_fraction, 
            random_state=42, 
            shuffle=True
        )

        # 4. Write Manifests
        write_manifest(train_meta, "train")
        write_manifest(valid_meta, "valid")
        
        print("\n--- DONE ---")
        print(f"Wavs location: {output_wav_dir}")
        print(f"Manifests: {manifest_dir}")
    else:
        print("No samples were processed successfully.")
