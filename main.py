import pickle
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# internal imports
from data_parser import run_data_parser, summarize_data_overview, convert_to_huggingface_dataset
from model_loader import prepare_resampled_dataloader, resample_to_16k, mask_spectrogram, plot_masked_dataset_statistics, load_data2vec_audio_model, run_model_on_dataset, train_self_supervised, evaluate_embeddings
from model_loader import *
from customize_model import *
from compute_stats import compute_cosine_similarity_matrix
from datasets import Dataset
import pandas as pd

NOVA_SAMPLES_PATH = ('/mnt5/noy/nova_samples/')
NOVAL_SINGLE_CHNL = NOVA_SAMPLES_PATH + 'single_chnl/'
NOVA_MULTI_CHNL = NOVA_SAMPLES_PATH + 'multi_chnl/'

def get_input_path_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', action='store_true', default="small", help='Use debug_chnl directory')
    parser.add_argument('--test', action='store_true', help='Use single_chnl directory')
    parser.add_argument('--batch_size', action='store_true', default=16, help='Size of training batch')
    parser.add_argument('--epoch', action='store_true', default=1, help='Number of epochs')
    parser.add_argument('--mask_ratio', action='store_true', default=0.15,  help='Masking ratio')

    args = parser.parse_args()

    if args.test_dir == 'small':
        samples_path = NOVA_SAMPLES_PATH + 'debug_chnl/'
    elif args.test_dir == 'medium':
        samples_path = NOVA_SAMPLES_PATH + 'one_chnl/'
    elif args.test_dir == 'large':
        samples_path = NOVA_SAMPLES_PATH + 'full_chnl/'
    return samples_path, args

if __name__ == '__main__':
    samples_path, parse_args = get_input_path_from_args()
    single_chnl_df = run_data_parser(samples_path)  # returns df
    #summarize_data_overview(single_chnl_df)
    model, feature_extractor, optimizer, device = load_data2vec_audio_model()
    dataset = prepare_masked_dataloader(single_chnl_df, interpolate_to_16k=False, mask_ratio=parse_args.mask_ratio, batch_size=parse_args.batch_size)
    # compute_cosine_similarity_matrix(dataset)  # optional
    # plot_masked_dataset_statistics(dataset, output_dir="plots/single_channel")
    train_feature_extractor_only(model, optimizer, dataset, device, parse_args.mask_ratio, parse_args.batch_size, parse_args.epoch)
    # features = run_model_on_dataset(dataset, model, model.feature_extractor, device)
