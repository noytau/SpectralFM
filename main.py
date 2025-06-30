import pickle
import pandas as pd
import matplotlib.pyplot as plt

# internal imports
from data_parser import run_data_parser, summarize_data_overview, convert_to_huggingface_dataset
from model_loader import mask_spectrogram, plot_masked_dataset_statistics, load_data2vec_audio_model, run_model_on_masked_dataset, train_self_supervised, evaluate_embeddings
from datasets import Dataset
import pandas as pd

NOVA_SAMPLES_PATH = ('/mnt5/noy/nova_samples/')
NOVAL_SINGLE_CHNL = NOVA_SAMPLES_PATH + 'single_chnl/'
NOVA_MULTI_CHNL = NOVA_SAMPLES_PATH + 'multi_chnl/'


if __name__ == '__main__':

    single_chnl_df = (run_data_parser()) # returns
    # summarize_data_overview(single_chnl_df) enable to create a summary of the dataset
    sinle_chnl_hf_dataset = convert_to_huggingface_dataset(single_chnl_df)
    print(f"First 5 rows of single channel dataset:\n{sinle_chnl_hf_dataset[:5]}")
    masked_dataset = mask_spectrogram(sinle_chnl_hf_dataset)
    print(f"First 5 rows of masked single channel dataset:\n{sinle_chnl_hf_dataset[:5]}") # dataset now contains 3 columns: data, masked_data, mask_indices
    # plot_masked_dataset_statistics(masked_dataset, output_dir="plots/single_channel")
    model, feature_extractor, device = load_data2vec_audio_model()
    del masked_dataset["mask_indices"]  # Remove mask_indices column for model input
    masked_dataset = Dataset.from_dict(masked_dataset)
    #features = run_model_on_masked_dataset(masked_dataset, model, model.feature_extractor, device)

    #train_self_supervised(model, model.feature_extractor, device, masked_dataset) #
    #evaluate_embeddings(model, model.feature_extractor, device, masked_dataset)
