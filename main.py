import pickle
import pandas as pd
import matplotlib.pyplot as plt

# internal imports
from data_parser import parse_directory_to_dict, inspect_chnl_spectograms, run_data_parser, summarize_data_overview
from model_loader import load_and_apply_audio_model

NOVA_SAMPLES_PATH = ('/mnt5/noy/nova_samples/')
NOVAL_SINGLE_CHNL = NOVA_SAMPLES_PATH + 'single_chnl/'
NOVA_MULTI_CHNL = NOVA_SAMPLES_PATH + 'multi_chnl/'


if __name__ == '__main__':

    summarize_data_overview(run_data_parser())
    #load_and_apply_audio_model()
