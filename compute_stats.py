import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Stats:
    """
    Used to plot graphs, collect statistics on data.
    """
    def __init__(self, df, output_dir="plots"):
        self.output_dir = output_dir
        self.df = df
        os.makedirs(output_dir, exist_ok=True) # make dir incase its missing

    def pass_dataset(self, dataset):
        self.dataset = dataset

    def pass_model(self, model):
        self.model = model

    def pass_embedding(self, embedding):
        self.embedding = embedding

    def pass_params(self, model, dataset, embedding):
        self.pass_dataset(dataset)
        self.pass_model(model)
        self.pass_embedding(embedding)

    def plot_dataset_stats(self):
        """
        Call plotting functions for data processing
        """
        self.plot_1d_spectrogram()
        self.plot_masked_spectrograms()
        self.scatterplot_mean_vs_std()
        self.compute_cosine_similarity_matrix()

    def compute_cosine_similarity_matrix(self, name="cosine_similarity_spectogram"):

        dataset = self.dataset
        # Get all data from the specified column
        data_tensor = torch.stack([sample["data"] for sample in dataset])

        # Normalize for cosine similarity
        normalized = torch.nn.functional.normalize(data_tensor, p=2, dim=1)

        # Compute cosine similarity matrix
        sim_matrix = torch.matmul(normalized, normalized.T).cpu().numpy()

        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(sim_matrix, cmap="viridis", xticklabels=False, yticklabels=False)
        plt.title(f"Cosine Similarity Heatmap ({name})")
        plt.xlabel("Sample Index")
        plt.ylabel("Sample Index")
        plt.tight_layout()
        #plt.show()
        #plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{name}.png"))
        plt.close()

        return sim_matrix

    def compute_cosine_similarity_matrix_from_embeddings(self, embeddings, name="cosine_similarity_embedding"):
        """
        embeddings: Tensor of shape [N, D], e.g. from model output (mean pooled or CLS token)
        """
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)

        # Normalize embeddings
        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Compute cosine similarity
        sim_matrix = torch.matmul(normalized, normalized.T).cpu().numpy()

        # Plot heatmap
        indices = list(range(embeddings.shape[0]))
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_matrix, cmap="viridis", xticklabels=False, yticklabels=False)
        plt.title("Cosine Similarity Heatmap (Embeddings)")
        plt.xlabel("Sample Index")
        plt.ylabel("Sample Index")
        plt.tight_layout()
        #plt.show()
        #plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{name}.png"))
        plt.close()

        return sim_matrix

    def plot_1d_spectrogram(self, title="1D Spectrogram", num_samples=5):
        """
        Plot sample 1D spectrograms from the DataFrame.
        Assumes each row is a 1D spectrogram.
        """

        for i in range(min(num_samples, len(self.df))):
            plt.figure()
            plt.plot(self.df.iloc[i].values)
            plt.title(f"{title} - Sample {i}")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plot_path = os.path.join(self.output_dir, f"sample_{i}.png")
            plt.savefig(plot_path)
            plt.close()

    # Additional utility functions
    def describe_dataset(self):
        """
        Print basic statistics for each column in the DataFrame.
        """
        print("DataFrame Statistics:")
        print(self.df.describe())

    def summarize_data_overview(self):
        """
        Display summary statistics, missing value counts, and sample plots.
        """
        # logger, log_path = setup_logger() # Uncomment to enable logging
        self.describe_dataset()
        self.plot_1d_spectrogram(self.df, num_samples=3)

    def plot_masked_spectrograms(self, num_samples=5):
        dataset = self.dataset
        data = [sample["data"] for sample in dataset]
        masked_data = [sample["masked_data"] for sample in dataset]
        for i in range(num_samples):
            plt.figure()
            plt.plot(data[i], label='Original')
            plt.plot(masked_data[i], label='Masked', linestyle='--')
            plt.title(f"Spectrogram {i}: Original vs Masked")
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, f"spectrogram_{i}_compare.png"))
            plt.close()

    def scatterplot_mean_vs_std(self, name="scatter_std_vs_mean_data.png"):
        dataset = self.dataset
        data = [sample["data"] for sample in dataset]
        plt.figure()
        plt.scatter(np.mean(data, axis=1), np.std(data, axis=1), s=10, alpha=0.7, color='blue', label='Original')
        plt.xlabel("Mean")
        plt.ylabel("Standard Deviation")
        plt.title("Std vs Mean (Original vs Masked)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, name))
        plt.close()

