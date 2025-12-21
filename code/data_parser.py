import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
from datasets import Dataset
import argparse
import torchaudio
import numpy as np
import torch


# Data parsing and inspection functions

def print_multi_channels_info(channel_dfs):
    info = {ch: df.shape[1] for ch, df in channel_dfs.items()}
    print(f"Number of active channels: {len(info)}")
    print("Channel details (channel: number of columns):")
    print(info)
    # print("Length of each channel's data:")
    # print("" + "\n".join([f"{ch}: {len(df)}" for ch, df in channel_dfs.items()]))
    # print("Length of first channel's data:")
    if channel_dfs:
        first_channel = next(iter(channel_dfs.values()))
        print(f"Length of first channel's data: {len(first_channel)}")

def inspect_chnl_spectograms(df, source=None):
    '''
    Load and parse dataset, handling both single and multi-channel data.
    '''
    df = df.drop(columns=['x', 'y'], errors='ignore')  # drop 'x' and 'y' columns if they exist)
    # print(f"Columns in dataset: {df.columns.tolist()}") # fixme debug
    channel_dfs = parse_dataset_to_dict(df)

    if source:
        print(f"File: {source}")
    if len(channel_dfs) == 1:
        print("Single channel data detected.")
    else:
        print("Multi-channel data detected.")
    return channel_dfs


def load_and_preview_dataset(csv_path, n_rows=5):
    df = pd.read_csv(csv_path)
    print(f"Loaded original dataset with shape: {df.shape}")
    return df

def parse_dataset_to_dict(df):
    '''

    :param df:
    :return: dictionary where key is channel number and value is a dataframe
    '''
    groups = {}
    # Get existing channels in this dataset
    # Extract and sort channel names, converting component_# to just the number
    channels = sorted(
        {
            col.split(":")[0] for col in df.columns
        },
        key=lambda x: (
            x.startswith("component_"),
            int(x.split("_")[-1]) if x.startswith("component_") else int(x)
        )
    )
    channels = [x.split("_")[-1] if x.startswith("component_") else x for x in channels]
    print(f"Channels found in dataset: {channels}") #fixme debug
    channel_dfs = {}
    for ch in channels:
        # Select columns for this channel
        ch_cols = [col for col in df.columns if col.startswith(f"{ch}:")]
        # Rename columns to just the number after ":"
        renamed_cols = [col.split(":")[1] for col in ch_cols]
        # Create new DataFrame
        channel_dfs[ch] = df[ch_cols].copy()
        channel_dfs[ch].columns = renamed_cols

    # Print info on dictionary
    print_multi_channels_info(channel_dfs)
    return channel_dfs


def parse_wav_directory_to_dict(directory):
    """
    Parse all .wav files in a directory into a list of dictionaries.

    Each item in the list contains:
    - 'data': the DataFrame loaded from the file (spectrogram)
    - 'source': the filename from which the DataFrame was loaded

    :param directory: Path to the directory containing .wav files.
    :return: List of dicts with keys 'data' and 'source'.
    """
    import os
    dataset = []

    import torchaudio
    import numpy as np
    import torch
    
    # Helper to create spectrogram from waveform
    def prepare_spectrogram(waveform, sample_rate=16000, n_mels=128):
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels
        )
        spectrogram = mel_spectrogram_transform(waveform)
        return spectrogram

    def load_wav_to_dataframe(wav_path, sample_rate=16000, n_mels=128):
        try:
            waveform, sr = torchaudio.load(wav_path)
            if sr != sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            
            # Ensure waveform is mono if it's stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            spectrogram = prepare_spectrogram(waveform, sample_rate, n_mels)
            # Convert spectrogram to DataFrame
            # Each row is a time step, each column is a mel band
            df = pd.DataFrame(spectrogram.squeeze(0).numpy())
            return df
        except Exception as e:
            print(f"Error loading or processing {wav_path}: {e}")
            return None

    dataset = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            df = load_wav_to_dataframe(file_path)
            if df is not None:
                dataset.append({
                    'data': df,
                    'source': filename
                })
        else:
            print(f"Skipping {filename}: not a .wav file")


    print(f"Parsed {len(dataset)} files from {directory}")  # fixme debug
    return dataset

def view_data_range(df, channel):
    """
    View the range of values in a specific channel of the DataFrame.

    :param df: DataFrame containing the data.
    :param channel: Channel to inspect.
    """
    if channel in df.columns:
        print(f"Range of values in channel {channel}: {df[channel].min()} to {df[channel].max()}")
    else:
        print(f"Channel {channel} not found in DataFrame.")



def run_data_parser(samples_path):
    # parse both directories
    # fixme divide this function into single and multi parser functions
    #multi_data = parse_directory_to_dict(NOVA_MULTI_CHNL)
    single_data = parse_wav_directory_to_dict(samples_path)

    # merge into one list if needed
    #combined_data = single_data + multi_data

    all_dfs = []

    for entry in single_data:
        df = entry['data']
        source = entry['source']
        channel_dfs = inspect_chnl_spectograms(df, source)
        all_dfs.extend(channel_dfs.values())

    # Combine all DataFrames into one large dataset
    final_df = pd.concat(all_dfs, ignore_index=True)
    #final_df = Dataset.from_pandas(final_df) # convert to HuggingFace Dataset format
    #dataset = final_df.map(lambda x: {"data": [x[f"f{i}"] for i in range(245)]}, remove_columns=final_df.column_names) # fixme remove
    #dataset = final_df.map(lambda x: {"data": [x[f"{i}"] for i in range(245)]}, remove_columns=final_df.column_names)
    print(f"Final combined dataset shape: {final_df.shape}") # merge all dfs together to one large df
    return final_df

def convert_to_huggingface_dataset(df):
    """
    Convert a pandas DataFrame to a Hugging Face Dataset with 'data' as a list of values.

    :param df: Pandas DataFrame to convert.
    :return: Hugging Face Dataset with a single column 'data'.
    """
    data_dict = {"data": df.values.tolist()}
    return Dataset.from_dict(data_dict)

# DataFrame normalization/standardization utilities

def normalize_dataframe(df):
    """
    Normalize DataFrame values to range [0, 1] column-wise.
    """
    return (df - df.min()) / (df.max() - df.min())

def standardize_dataframe(df):
    """
    Standardize DataFrame values to zero mean and unit variance column-wise.
    """
    return (df - df.mean()) / df.std()


def count_missing_values(df):
    """
    Print the number of missing (NaN) values in each column.
    """
    print("Missing values per column:")
    print(df.isnull().sum())


# Example logger setup utility
def setup_logger(log_dir="logs", prefix="run"):
    """
    Set up a logger that writes to a file with a timestamp in the filename.
    Returns the logger and the log filename.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{prefix}_{timestamp}.log")
    logger = logging.getLogger(prefix)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger, log_filename


