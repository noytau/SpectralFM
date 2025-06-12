import pickle
import pandas as pd
import matplotlib.pyplot as plt

NOVA_SAMPLES_PATH = ('/mnt5/noy/nova_samples/')
NOVAL_SINGLE_CHNL = NOVA_SAMPLES_PATH + 'single_chnl/'
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

def inspect_multi_chnl_spectograms(df):
    '''
    load and parse dataset
    '''
    df = df.iloc[:,2:] # first two columns in dataset are always empty
    # fixme Check if data set is single channel or multi-channel
    parse_dataset_to_dict(df)

def load_and_preview_dataset(csv_path, n_rows=5): # fixme noy: use this for singular channel
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
    #channels = sorted({col.split(":")[0] for col in df.columns}, key=int)
    channels = sorted(
        {col.split(":")[0] for col in df.columns},
        key=lambda x: (x.startswith("component_"), int(x.split("_")[-1]) if x.startswith("component_") else int(x))
    )

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
    Parse all .pkl files in a directory into a dictionary.

    :param directory: Path to the directory containing .pkl files.
    :return: Dictionary where keys are filenames and values are DataFrames.
    """
    import os
    data_dict = {}

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            print(f"Loading {filename}...") # fixme debug
            data = pickle.load(f)
            if isinstance(data, pd.DataFrame):
                data_dict[filename] = data
            else:
                print(f"Skipping {filename}: not a DataFrame")

    print(f"Parsed {len(data_dict)} files from {directory}") # fixme debug
    return data_dict

