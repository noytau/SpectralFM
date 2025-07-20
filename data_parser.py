import pickle
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
from datasets import Dataset

NOVA_SAMPLES_PATH = ('/mnt5/noy/nova_samples/')
NOVAL_SINGLE_CHNL = NOVA_SAMPLES_PATH + 'debug_chnl/'
#NOVAL_SINGLE_CHNL = NOVA_SAMPLES_PATH + 'single_chnl/'
NOVA_MULTI_CHNL = NOVA_SAMPLES_PATH + 'multi_chnl/'

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

def inspect_pkl(path, img_shape=(28, 28)): # Used for initial dataset with no mixed channels
    with open(path, 'rb') as f:
        data = pickle.load(f)

    print(f"Type: {type(data)}")

    if isinstance(data, pd.DataFrame):
        print(f"Shape: {data.shape}")
        print("Previewing first 5 images:")

        for i in range(5):
            img_array = data.iloc[i].values.reshape(img_shape)
            plt.figure()
            plt.imshow(img_array, cmap='gray')
            plt.title(f"Image {i}")
            plt.axis('off')
            plt.show()


    elif isinstance(data, dict):
        print("Preview (dict):")
        for i, (k, v) in enumerate(data.items()):
            print(f"{k}: {v}")
            if i >= 4:
                break
    elif isinstance(data, list):
        print("Preview (list):")
        print(data[:5])
    else:
        print("Preview:")
        print(data)


def parse_directory_to_dict(directory):
    """
    Parse all .pkl files in a directory into a list of dictionaries.

    Each item in the list contains:
    - 'data': the DataFrame loaded from the file
    - 'source': the filename from which the DataFrame was loaded

    :param directory: Path to the directory containing .pkl files.
    :return: List of dicts with keys 'data' and 'source'.
    """
    import os
    dataset = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, pd.DataFrame):
                dataset.append({
                    'data': data,
                    'source': filename
                })
            else:
                print(f"Skipping {filename}: not a DataFrame")

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



def run_data_parser():
    # parse both directories
    # fixme divide this function into single and multi parser functions
    #multi_data = parse_directory_to_dict(NOVA_MULTI_CHNL)
    single_data = parse_directory_to_dict(NOVAL_SINGLE_CHNL)

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


# Additional utility functions
def describe_dataset(df):
    """
    Print basic statistics for each column in the DataFrame.
    """
    print("DataFrame Statistics:")
    print(df.describe())

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

def plot_1d_spectrogram(df, title="1D Spectrogram", num_samples=5):
    """
    Plot sample 1D spectrograms from the DataFrame.
    Assumes each row is a 1D spectrogram.
    """
    import matplotlib.pyplot as plt
    import os

    plot_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for i in range(min(num_samples, len(df))):
        plt.figure()
        plt.plot(df.iloc[i].values)
        plt.title(f"{title} - Sample {i}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plot_path = os.path.join(plot_dir, f"sample_{i}.png")
        plt.savefig(plot_path)
        plt.close()

def summarize_data_overview(df):
    """
    Display summary statistics, missing value counts, and sample plots.
    """
    # logger, log_path = setup_logger() # Uncomment to enable logging
    describe_dataset(df)
    plot_1d_spectrogram(df, num_samples=3)
