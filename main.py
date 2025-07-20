import pickle
import pandas as pd
import matplotlib.pyplot as plt

# internal imports
from data_parser import run_data_parser, summarize_data_overview, convert_to_huggingface_dataset
from model_loader import prepare_resampled_dataloader, resample_to_16k, mask_spectrogram, plot_masked_dataset_statistics, load_data2vec_audio_model, run_model_on_dataset, train_self_supervised, evaluate_embeddings
from compute_stats import compute_cosine_similarity_matrix
from datasets import Dataset
import pandas as pd

NOVA_SAMPLES_PATH = ('/mnt5/noy/nova_samples/')
NOVAL_SINGLE_CHNL = NOVA_SAMPLES_PATH + 'single_chnl/'
NOVA_MULTI_CHNL = NOVA_SAMPLES_PATH + 'multi_chnl/'


if __name__ == '__main__':

    single_chnl_df = (run_data_parser()) # returns
    # summarize_data_overview(single_chnl_df) enable to create a summary of the dataset

    model, feature_extractor, device = load_data2vec_audio_model()
    dataset = prepare_resampled_dataloader(single_chnl_df)
    compute_cosine_similarity_matrix(dataset)  # Compute and plot cosine similarity matrix
    # plot_masked_dataset_statistics(masked_dataset, output_dir="plots/single_channel")

    features = run_model_on_dataset(dataset, model, model.feature_extractor, device)
