import pickle
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datasets import Dataset
import pandas as pd

# internal imports
from data_parser import *
from model_loader import *
from compute_stats import *

NOVA_SAMPLES_PATH = ('/mnt5/noy/nova_samples/')
NOVAL_SINGLE_CHNL = NOVA_SAMPLES_PATH + 'single_chnl/'
NOVA_MULTI_CHNL = NOVA_SAMPLES_PATH + 'multi_chnl/'

def get_input_path_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=1, help='Run configuration index') # fixme maybe pass somewhere meaningful?
    parser.add_argument('--test_dir', type=str, default="small", help='Path to sample dir')
    parser.add_argument('--batch_size', type=int, default=16, help='Size of training batch')
    parser.add_argument('--epoch', type=int, default=1, help='Number of epochs')
    parser.add_argument('--mask_ratio',type=float, default=0.15,  help='Masking ratio')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--loss_function', type=str, default="mse", help='Loss function for training')

    args = parser.parse_args()

    if args.test_dir == 'small':
        samples_path = NOVA_SAMPLES_PATH + 'debug_chnl/'
    elif args.test_dir == 'medium':
        samples_path = NOVA_SAMPLES_PATH + 'one_chnl/'
    elif args.test_dir == 'large':
        samples_path = NOVA_SAMPLES_PATH + 'full_chnl/'
    else: # if not stated, use debug
        samples_path = NOVA_SAMPLES_PATH + 'debug_chnl/'
    return samples_path, args

if __name__ == '__main__':
    samples_path, parse_args = get_input_path_from_args()
    single_chnl_df = run_data_parser(samples_path)  # returns df
    # init stats class to plot and compute data
    # stats = Stats(df=single_chnl_df, argparse=parse_args)
    #summarize_data_overview(single_chnl_df)
    model, feature_extractor, optimizer, device = load_custom_data2vec_audio_model(parse_args.learning_rate)
    dataloader, masked_dataset = prepare_masked_dataloader(single_chnl_df, interpolate_to_16k=False, mask_ratio=parse_args.mask_ratio, batch_size=parse_args.batch_size)
