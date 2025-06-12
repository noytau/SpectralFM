import pickle
import pandas as pd
import matplotlib.pyplot as plt

# internal imports
from data_parser import parse_directory_to_dict, inspect_multi_chnl_spectograms

NOVA_SAMPLES_PATH = ('/mnt5/noy/nova_samples/')
NOVAL_SINGLE_CHNL = NOVA_SAMPLES_PATH + 'single_chnl/'
NOVA_MULTI_CHNL = NOVA_SAMPLES_PATH + 'multi_chnl/'


if __name__ == '__main__':

    # fixme should it be one caller function that calls both?
    # parse both directories
    multi_data = parse_directory_to_dict(NOVA_MULTI_CHNL)
    single_data = parse_directory_to_dict(NOVAL_SINGLE_CHNL)


    # merge into one dict if needed
    combined_data = {
        "single": single_data,
        "multi": multi_data
    }

    for filename, df in combined_data["multi"].items():
        print(f"File: {filename}, Shape: {df.shape}")
        inspect_multi_chnl_spectograms(df)
