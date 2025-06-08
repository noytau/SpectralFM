# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from numpy.ma.core import shape

NOVA_SAMPLES_PATH = ('/Users/noyhassid/Documents/NovaData/') # fixme noy change to server dir
NOVAL_PKL_FIL = NOVA_SAMPLES_PATH + 'spectra0000_batch5.pkl'
NOVA_DATASET = NOVA_SAMPLES_PATH + '/mixed channels/dataset0002'


def print_channels_dict_info(channel_dfs):
    info = {ch: df.shape[1] for ch, df in channel_dfs.items()}
    print(f"Number of active channels: {len(info)}")
    print("Channel details (channel: number of columns):")
    print(info)

def inspect_spectrogram_pkl(path):
    '''
    load and parse dataset
    '''
    with open(path, 'rb') as f:
        df = pickle.load(f)
    df = df.iloc[:,2:] # first two columns in dataset are always empty

    print(f"Original data shape: {df.shape}")
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
    channels = sorted({col.split(":")[0] for col in df.columns}, key=int)

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
    print_channels_dict_info(channel_dfs)
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
        print(data)# # U


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    inspect_spectrogram_pkl(NOVA_DATASET)
    #load_and_preview_dataset(NOVA_DATASET) # fixme for singular data channel
