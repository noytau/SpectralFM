import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import seaborn as sns
import umap
import matplotlib.pyplot as plt

# internal imports
from data_parser import *
from model_loader import *

PLOTS_DIR = "/mnt5/noy/code/plots"

class Stats:
    """
    Used to plot graphs, collect statistics on data.
    """
    def __init__(self, df, argparse=None):
        import datetime

        self.df = df
        date_str = datetime.datetime.now().date().isoformat()
        if argparse:
            experiment_string = argparse.model_path.split('/')[-1]
            experiment_string = "experiment-" + experiment_string # fixme noy consider including len of dataset
        else:
            experiment_string = f"experiment-{date_str}"
        output_dir = os.path.join(PLOTS_DIR, experiment_string)
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True) # make dir incase its missing
        print(f"Output directory for plots: {output_dir}")

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

    # Post model evaluate phase
    def plot_model_stats(self, model, pre_train_embeddings, post_train_embeddings, original_model_embeddings):
        self.model = model
        self.pre_train_embeddings = pre_train_embeddings
        self.post_train_embeddings = post_train_embeddings
        self.original_model_embeddings = original_model_embeddings

        # Get top-k neighbors for every sample
        similar_spectograms = self.get_top_m_with_k_similar_fast(m=len(self.dataset), k=10, method='cosine')

        # Get selected refs from export_similarity_l2_stats and filter for plotting
        selected_refs = self.export_similarity_l2_stats(similar_spectograms, method='cosine', filename="similarity_l2_stats.csv")
        self.plot_embeddings_with_similar_highlighted(pre_train_embeddings, post_train_embeddings, original_model_embeddings, selected_refs)
        self.plot_similar_spectrograms(selected_refs)


        #selected_list = self.extract_grouped_refs(selected_refs)

        # cosing similarities
        #for neighbors in selected_list:
        #    self.compute_cosine_similarity_matrix(neighbors)
        #    self.compute_cosine_similarity_matrix_from_embeddings(neighbors, original_model_embeddings, name="cosine_similarity_original_model_embeddings")
        #    self.compute_cosine_similarity_matrix_from_embeddings(neighbors, pre_train_embeddings, name="cosine_similarity_pre_train_embeddings")
        #    self.compute_cosine_similarity_matrix_from_embeddings(neighbors, post_train_embeddings, name="cosine_similarity_post_train_embeddings")

    def extract_grouped_refs(self, selected_refs):
        """
        From selected_refs {ref_idx: [(neighbor_idx, score), ...]},
        create a list of lists where each list starts with the ref_idx followed by its neighbors.
        Example: [[0, 5, 8], [1, 3]]
        """
        return [
            [ref_idx] + [neighbor_idx for neighbor_idx, _ in neighbors]
            for ref_idx, neighbors in selected_refs.items()
        ]

    def compute_cosine_similarity_matrix(self, selected_indices=None, name="cosine_similarity_spectogram"):
        if selected_indices is None:
            selected_indices = list(range(len(self.dataset)))
        dataset = [self.dataset[i] for i in selected_indices]
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

    def compute_cosine_similarity_matrix_from_embeddings(self, embeddings, selected_indices=None, name="cosine_similarity_embedding"):
        """
        embeddings: Tensor of shape [N, D], e.g. from model output (mean pooled or CLS token)
        """
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)

        if selected_indices is None:
            selected_indices = list(range(embeddings.shape[0]))

        selected_indices = torch.tensor(selected_indices, dtype=torch.long)
        embeddings = embeddings[selected_indices]

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

    def get_top_m_with_k_similar_fast(self, m=5, k=5, method='cosine'):
        data = [sample["data"] for sample in self.dataset]
        N = len(data)
        # Flatten
        flat_dataset = torch.stack(data).numpy()

        # Select m reference indices (simple strategy: evenly spaced)
        step = max(1, N // m)
        reference_indices = list(range(0, N, step))[:m]

        # Fit NearestNeighbors
        metric = 'cosine' if method == 'cosine' else 'euclidean'
        nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
        nn.fit(flat_dataset)

        results = {}
        for idx in reference_indices:
            query = flat_dataset[idx].reshape(1, -1)
            distances, indices = nn.kneighbors(query, n_neighbors=k + 1)

            neighbor_scores = []
            for neighbor_idx, dist in zip(indices[0], distances[0]):
                if neighbor_idx == idx:
                    continue  # skip self
                if method == 'cosine':
                    score = 1 - dist  # cosine similarity
                else:  # euclidean
                    score = -dist  # lower distance = more similar
                neighbor_scores.append((neighbor_idx, score))
                if len(neighbor_scores) == k:
                    break

            results[idx] = neighbor_scores

        return results

    import matplotlib.pyplot as plt

    def plot_similar_spectrograms(self, results):
        """
        Plot each reference spectrogram and its k most similar neighbors
        with cosine similarity scores.

        Parameters:
        - spectrogram_list: list of torch.Tensor, each of shape (245,)
        - results: dict {reference_idx: [(neighbor_idx, score), ...]}
        """
        spectrogram_list = [sample["data"] for sample in self.dataset]

        for ref_idx, neighbor_tuples in results.items():
            k = len(neighbor_tuples)
            fig, axs = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3), squeeze=False)

            # Plot reference
            ref_signal = spectrogram_list[ref_idx].cpu().numpy()
            axs[0, 0].plot(ref_signal)
            axs[0, 0].set_title(f"Reference (idx={ref_idx})")
            axs[0, 0].set_xlabel("Time")
            axs[0, 0].set_ylabel("Amplitude")

            # Plot each neighbor with score
            for j, (neighbor_idx, score) in enumerate(neighbor_tuples):
                neighbor_signal = spectrogram_list[neighbor_idx].cpu().numpy()
                axs[0, j + 1].plot(neighbor_signal)
                axs[0, j + 1].set_title(f"Similar {j + 1}\nidx={neighbor_idx} | score={score:.2f}")
                axs[0, j + 1].set_xlabel("Time")

            plt.tight_layout()
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, f"spectrogram_{ref_idx}_similar_spectograms.png"))
            plt.close()

    def extract_embeddings_from_results_fn(self, embeddings, results, device='cpu'):

        """
        Given precomputed embeddings and a spectrogram dictionary, return a structured
        dict of embeddings for reference and neighbor spectrograms.

        Parameters:
        - embeddings: Tensor of shape [N, D], containing embedding vectors
        - results: dict {ref_idx: [(neighbor_idx, input_score), ...]}
        - device: 'cuda' or 'cpu'

        Returns:
        - dict: {ref_idx: (ref_embedding, [(neighbor_embedding, input_score), ...])}
        """

        embedding_results = {}

        for ref_idx, neighbors in results.items():
            ref_emb = embeddings[ref_idx].cpu()

            neighbor_tuples = []
            for neighbor_idx, input_score in neighbors:
                neighbor_emb = embeddings[neighbor_idx].cpu()
                neighbor_tuples.append((neighbor_emb, input_score))

            embedding_results[ref_idx] = (ref_emb, neighbor_tuples)

        return embedding_results

    def visualize_embeddings_2D(self, emb1, emb2):
        data = torch.stack([emb1, emb2]).numpy()

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(data)

        plt.figure(figsize=(5, 5))
        plt.scatter(reduced[:, 0], reduced[:, 1], color=["blue", "orange"])
        plt.text(reduced[0, 0], reduced[0, 1], "emb1", fontsize=12)
        plt.text(reduced[1, 0], reduced[1, 1], "emb2", fontsize=12)
        plt.title("PCA Projection of Two Embeddings")
        plt.grid(True)
        plt.axis("equal")
        plt.show()

    def cluster_embeddings(self, embeddings, n_clusters=5):
        """
        Clusters the embeddings using KMeans and visualizes the clusters.

        Parameters:
        - embeddings: Tensor of shape [N, D], containing embedding vectors
        - n_clusters: Number of clusters to form

        Returns:
        - labels: Cluster labels for each embedding
        """
        from sklearn.cluster import KMeans
        from sklearn.manifold import TSNE

        embeddings_np = embeddings.cpu().numpy()

        # Step 1: Cluster in original space
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings_np)

        # Step 2: Optional PCA to speed up t-SNE
        pca = PCA(n_components=50)
        pca_embeddings = pca.fit_transform(embeddings_np)

        # Step 3: t-SNE for visualization
        tsne_embeddings = TSNE(n_components=2, random_state=42).fit_transform(pca_embeddings)

        # Step 3: UMAP for visualization
        umap_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(pca_embeddings)

        # Step 4: Plot
        palette = sns.color_palette("hls", n_colors=n_clusters)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], hue=labels, palette=palette, s=50, legend=False)
        plt.title(f"KMeans Clustering (on raw embeddings), Visualized in t-SNE — n={n_clusters}")
        plt.xlabel("tSNE-1")
        plt.ylabel("tSNE-2")
        #plt.legend(title="Cluster")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"kmeans_clustered_then_tsne_n{n_clusters}.png"))
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1], hue=labels, palette=palette, s=50, legend=False)
        plt.title(f"KMeans Clustering (on raw embeddings), Visualized in UMAP — n={n_clusters}")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        #plt.legend(title="Cluster")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"kmeans_clustered_then_umap_n{n_clusters}.png"))
        plt.close()

    def compare_clustering_algorithms(self, embeddings, n_clusters=5):
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
        from sklearn.mixture import GaussianMixture
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os

        import umap
        from sklearn.manifold import TSNE

        embeddings_np = embeddings.cpu().numpy()

        # Optional PCA to 50 dims
        pca = PCA(n_components=50)
        embeddings_pca = pca.fit_transform(embeddings)

        # Reduce to 2D for visualization
        tsne = TSNE(n_components=2, random_state=42)
        tsne_embeddings = tsne.fit_transform(embeddings_pca)

        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        umap_embeddings = umap_model.fit_transform(embeddings_pca)

        algorithms = {
            "KMeans": KMeans(n_clusters=n_clusters, random_state=42),
            "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
            "Spectral": SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42),
            "GMM": GaussianMixture(n_components=n_clusters, random_state=42),
            "DBSCAN": DBSCAN(eps=0.1, min_samples=5)
        }

        for name, algo in algorithms.items():
            if name == "GMM":
                labels = algo.fit_predict(embeddings)
            else:
                labels = algo.fit(embeddings).labels_

            # Optional: compute silhouette if labels are valid
            try:
                score = silhouette_score(embeddings, labels)
                print(f"[{name}] Silhouette score: {score:.4f}")
            except:
                print(f"[{name}] Silhouette score: could not be computed")

            for method, reduced in zip(["tsne", "umap"], [tsne_embeddings, umap_embeddings]):
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="hls", s=50, legend=False)
                plt.title(f"{name} Clustering visualized with {method.upper()}")
                plt.xlabel(f"{method.upper()}-1")
                plt.ylabel(f"{method.upper()}-2")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"{name.lower()}_{method}_clustering.png"))
                plt.close()

    def find_optimal_k(self, embeddings, k_range=range(2, 15)):
        embeddings_np = embeddings.cpu().numpy()
        best_k = None
        best_score = -1
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embeddings_np)
            score = silhouette_score(embeddings_np, labels)
            print(f"n_clusters={k}, silhouette_score={score:.4f}")
            if score > best_score:
                best_k = k
                best_score = score
        print(f"✅ Best k: {best_k} (score: {best_score:.4f})")
        return best_k

    def compare_and_visualize_embeddings(self, embedding_results):
        """
        For each reference embedding, compares it to all of its neighbors
        and visualizes similarity.

        Parameters:
        - embedding_results: dict of {ref_idx: (ref_embedding, [(neighbor_emb, score), ...])}
        """
        import torch.nn.functional as F

        for ref_idx, (ref_emb, neighbor_list) in embedding_results.items():
            # ref_emb: tensor, neighbor_list: list of (neighbor_emb, score)
            for i, (neighbor_emb, input_score) in enumerate(neighbor_list):
                cos_sim = F.cosine_similarity(ref_emb.unsqueeze(0), neighbor_emb.unsqueeze(0)).item()
                title = f"Ref {ref_idx} vs Neighbor {i} | CosSim: {cos_sim:.2f} | Score: {input_score:.2f}"
                self.visualize_embeddings_2D(ref_emb, neighbor_emb)


    def compute_l2_distances(self, ref_idx, neighbors, pre_embeddings_np, post_embeddings_np, original_model_embeddings_np):
        from numpy.linalg import norm

        ref_pre = pre_embeddings_np[ref_idx]
        ref_post = post_embeddings_np[ref_idx]
        ref_original = original_model_embeddings_np[ref_idx]

        dists_all_pre = [norm(x - ref_pre) for x in pre_embeddings_np]
        dists_all_post = [norm(x - ref_post) for x in post_embeddings_np]
        dists_all_orig = [norm(x - ref_original) for x in original_model_embeddings_np]

        avg_l2_all_pre = np.mean(dists_all_pre)
        avg_l2_all_post = np.mean(dists_all_post)
        avg_l2_all_orig = np.mean(dists_all_orig)

        neighbor_indices = [idx for idx, _ in neighbors]
        dists_neighbors_pre = [norm(pre_embeddings_np[idx] - ref_pre) for idx in neighbor_indices]
        dists_neighbors_post = [norm(post_embeddings_np[idx] - ref_post) for idx in neighbor_indices]
        dists_neighbors_orig = [norm(original_model_embeddings_np[idx] - ref_original) for idx in neighbor_indices]

        avg_l2_neighbors_pre = np.mean(dists_neighbors_pre)
        avg_l2_neighbors_post = np.mean(dists_neighbors_post)
        avg_l2_neighbors_orig = np.mean(dists_neighbors_orig)

        return avg_l2_all_pre, avg_l2_all_post, avg_l2_all_orig, avg_l2_neighbors_pre, avg_l2_neighbors_post, avg_l2_neighbors_orig

    def plot_embeddings_with_similar_highlighted(self, pre_train_embeddings, post_train_embeddings, original_model_embeddings, results):
        """
        For each reference in `results`, plot a PCA projection of all embeddings,
        highlighting:
          - The reference embedding (orange)
          - Its similar neighbors (purple) with index and similarity score
          - All others (gray)

        Parameters:
        - embeddings: torch.Tensor of shape (N, D)
        - results: dict {ref_idx: [(neighbor_idx, similarity_score), ...]}
        """

        post_embeddings_np = post_train_embeddings.cpu().numpy()
        pre_embeddings_np = pre_train_embeddings.cpu().numpy()
        original_embeddings_np = original_model_embeddings.cpu().numpy()
        pca_post = PCA(n_components=2)
        reduced_post = pca_post.fit_transform(post_embeddings_np)
        pca_pre = PCA(n_components=2)
        reduced_pre = pca_pre.fit_transform(pre_embeddings_np)
        pca_original = PCA(n_components=2)
        reduced_original = pca_original.fit_transform(original_embeddings_np)
        for ref_idx, neighbors in results.items():
            N = post_embeddings_np.shape[0]
            from matplotlib.cm import get_cmap
            cmap = get_cmap("winter")  # blue to green
            colors = ['gray'] * N
            labels = ['other'] * N
            scatter_colors_post = ['gray'] * N
            scatter_colors_pre = ['gray'] * N
            scatter_colors_original = ['gray'] * N

            colors[ref_idx] = 'red'
            labels[ref_idx] = 'reference'
            scatter_colors_post[ref_idx] = 'red'
            scatter_colors_pre[ref_idx] = 'red'
            scatter_colors_original[ref_idx] = 'red'

            for i, (neighbor_idx, _) in enumerate(neighbors):
                color = cmap(i / max(1, len(neighbors) - 1))
                colors[neighbor_idx] = color
                labels[neighbor_idx] = f'similar{i}'
                scatter_colors_post[neighbor_idx] = color
                scatter_colors_pre[neighbor_idx] = color
                scatter_colors_original[neighbor_idx] = color

            # Calculate L2 distances using the new method
            avg_l2_all_pre, avg_l2_all_post, avg_l2_all_original, avg_l2_neighbors_pre, avg_l2_neighbors_post, avg_l2_neighbors_original = self.compute_l2_distances(
                ref_idx, neighbors, pre_embeddings_np, post_embeddings_np, original_embeddings_np
            )

            # Set figure DPI for higher resolution
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

            # Define base marker size
            base_size = 40

            # Plot all points first
            ax.scatter(reduced_post[:, 0], reduced_post[:, 1], color=scatter_colors_post, alpha=0.5, s=base_size, marker='o', label='post')
            ax.scatter(reduced_pre[:, 0], reduced_pre[:, 1], color=scatter_colors_pre, alpha=0.8, s=base_size, marker='x', label='pre')
            ax.scatter(reduced_original[:, 0], reduced_original[:, 1], color=scatter_colors_original, alpha=0.8, s=base_size, marker='s', label='original')

            # Re-plot reference and neighbors on top with larger markers
            ax.scatter(reduced_post[ref_idx, 0], reduced_post[ref_idx, 1], color='red', alpha=0.9, s=base_size * 1.2, marker='o', edgecolor='black', linewidth=0.8, zorder=3)
            ax.scatter(reduced_pre[ref_idx, 0], reduced_pre[ref_idx, 1], color='red', alpha=1.0, s=base_size * 1.2, marker='x', edgecolor='black', linewidth=0.8, zorder=3)
            ax.scatter(reduced_original[ref_idx, 0], reduced_original[ref_idx, 1], color='red', alpha=1.0, s=base_size * 1.2, marker='s', edgecolor='black', linewidth=0.8, zorder=3)

            for i, (neighbor_idx, _) in enumerate(neighbors):
                color = cmap(i / max(1, len(neighbors) - 1))
                ax.scatter(reduced_post[neighbor_idx, 0], reduced_post[neighbor_idx, 1], color=color, alpha=0.9, s=base_size * 1.2, marker='o', edgecolor='black', linewidth=0.8, zorder=3)
                ax.scatter(reduced_pre[neighbor_idx, 0], reduced_pre[neighbor_idx, 1], color=color, alpha=1.0, s=base_size * 1.2, marker='x', edgecolor='black', linewidth=0.8, zorder=3)
                ax.scatter(reduced_original[neighbor_idx, 0], reduced_original[neighbor_idx, 1], color=color, alpha=1.0, s=base_size * 1.2, marker='s', edgecolor='black', linewidth=0.8, zorder=3)

            # Add table on the right side
            table_data = [(f"{idx}", f"{score:.2f}") for idx, score in neighbors]
            col_labels = ["Neighbor idx", "Score"]
            table = plt.table(cellText=table_data,
                              colLabels=col_labels,
                              cellLoc='center',
                              colWidths=[0.15, 0.15],
                              loc='right',
                              bbox=[1.05, 0.2, 0.3, 0.6])
            table.auto_set_font_size(False)
            table.set_fontsize(10)

            ax.set_title(f"PCA of Embeddings — Reference Index: {ref_idx}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            # Display distance statistics directly on the graph, positioned below the right-side legend table
            dist_text = (
                f"Avg L2 All:\n"
                f"  Original: {avg_l2_all_original:.2f}  "
                f"| Pre: {avg_l2_all_pre:.2f}  "
                f"| Post: {avg_l2_all_post:.2f}\n"
                f"Avg L2 Neighbors:\n"
                f"  Original: {avg_l2_neighbors_original:.2f}  "
                f"| Pre: {avg_l2_neighbors_pre:.2f}  "
                f"| Post: {avg_l2_neighbors_post:.2f}"
            )
            ax.text(1.05, -0.2, dist_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.8))
            ax.grid(True)
            plt.tight_layout()
            #plt.show()
            plt.savefig(os.path.join(self.output_dir, f"embedding{ref_idx}_similar_highlighted_after_training.png"))
            plt.close()

    def export_similarity_l2_stats(self, results, method='cosine', filename="similarity_l2_stats.csv"):
        """
        For each sample in the dataset, finds its k most similar neighbors (by the given method),
        computes average L2 distances (to all, to neighbors) for pre- and post-train embeddings,
        and saves the statistics to a CSV. Prints summary statistics.
        """
        import pandas as pd
        from numpy.linalg import norm

        original_model_embeddings_np = self.original_model_embeddings.cpu().numpy()
        pre_embeddings_np = self.pre_train_embeddings.cpu().numpy()
        post_embeddings_np = self.post_train_embeddings.cpu().numpy()

        rows = []
        for ref_idx, neighbors in results.items():
            avg_l2_all_pre, avg_l2_all_post, avg_l2_all_orig, avg_l2_neighbors_pre, avg_l2_neighbors_post, avg_l2_neighbors_orig = self.compute_l2_distances(
                ref_idx, neighbors, pre_embeddings_np, post_embeddings_np, original_model_embeddings_np
            )
            rows.append({
                "ref_idx": ref_idx,
                "avg_l2_all_pre": avg_l2_all_pre,
                "avg_l2_all_post": avg_l2_all_post,
                "avg_l2_all_orig": avg_l2_all_orig,
                "avg_l2_neighbors_pre": avg_l2_neighbors_pre,
                "avg_l2_neighbors_post": avg_l2_neighbors_post,
                "avg_l2_neighbors_orig": avg_l2_neighbors_orig
            })

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_dir, filename)
        df.to_csv(csv_path, index=False)
        print(f"Saved L2 similarity stats to {csv_path}")
        print(df.describe())

        # Calculate absolute difference between pre and post training L2 distances to neighbors
        df["abs_diff_pre_post"] = np.abs(df["avg_l2_neighbors_post"] - df["avg_l2_neighbors_pre"])

        # Sort by difference
        sorted_df = df.sort_values("abs_diff_pre_post")

        # Bottom 5 (smallest changes)
        bottom_refs = sorted_df.head(5)["ref_idx"].tolist()

        # Top 5 (largest changes)
        top_refs = sorted_df.tail(5)["ref_idx"].tolist()

        # Median 5 (around the middle)
        median_start = len(sorted_df) // 2 - 2
        median_refs = sorted_df.iloc[median_start:median_start + 5]["ref_idx"].tolist()

        print("Top 5 ref_idx (largest Δ):", top_refs)
        print("Bottom 5 ref_idx (smallest Δ):", bottom_refs)
        print("Median 5 ref_idx (middle Δ):", median_refs)

        # Also calculate absolute difference between original and post training L2 distances
        df["abs_diff_orig_post"] = np.abs(df["avg_l2_neighbors_post"] - df["avg_l2_neighbors_orig"])

        # Sort by difference
        sorted_df_orig_post = df.sort_values("abs_diff_orig_post")

        # Bottom 5 (smallest changes)
        bottom_refs_orig_post = sorted_df_orig_post.head(5)["ref_idx"].tolist()

        # Top 5 (largest changes)
        top_refs_orig_post = sorted_df_orig_post.tail(5)["ref_idx"].tolist()

        # Median 5 (around the middle)
        median_start_orig_post = len(sorted_df_orig_post) // 2 - 2
        median_refs_orig_post = sorted_df_orig_post.iloc[median_start_orig_post:median_start_orig_post + 5][
            "ref_idx"].tolist()

        print("Top 5 ref_idx (largest Δ orig-post):", top_refs_orig_post)
        print("Bottom 5 ref_idx (smallest Δ orig-post):", bottom_refs_orig_post)
        print("Median 5 ref_idx (middle Δ orig-post):", median_refs_orig_post)

        selected_ref_indices = list(set(
            top_refs + bottom_refs + median_refs +
            top_refs_orig_post + bottom_refs_orig_post + median_refs_orig_post
        ))
        selected_refs_dict = {ref_idx: results[ref_idx] for ref_idx in selected_ref_indices}
        return selected_refs_dict
def get_input_path_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/mnt5/noy/code/weights/experiment-mask=0.3-epoch=1_batch=16_loss_fn=mse_datalen=1000000_model_before_training.pt', help='Path to saved model')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Size of training batch')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='Masking ratio')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parse_args = get_input_path_from_args()
    samples_path = NOVA_SAMPLES_PATH + 'debug_chnl/' # fixme!
    single_chnl_df = run_data_parser(samples_path)  # returns df
    # init stats class to plot and compute data
    stats = Stats(df=single_chnl_df, argparse=parse_args)

    model, feature_extractor, optimizer, device = load_custom_data2vec_audio_model(parse_args.learning_rate)
    dataloader, masked_dataset = prepare_masked_dataloader(single_chnl_df, interpolate_to_16k=False, mask_ratio=parse_args.mask_ratio, batch_size=parse_args.batch_size)

    stretched_dataloader, stretched_masked_dataset = prepare_masked_dataloader(single_chnl_df, interpolate_to_16k=True, mask_ratio=parse_args.mask_ratio, batch_size=parse_args.batch_size)
    original_model, original_feature_extractor, _, _ = load_custom_data2vec_audio_model()

    stats.pass_dataset(masked_dataset)
    stats.plot_dataset_stats()

    pre_train_outputs, pre_train_embeddings = evaluate_embedding_from_model(model, dataloader, device, parse_args.batch_size)
    original_model_outputs, original_model_embeddings = evaluate_embedding_from_model(original_model, stretched_dataloader, device, parse_args.batch_size)

    post_train_outputs, post_train_embeddings = evaluate_embedding_from_model(model, dataloader, device, batch_size=parse_args.batch_size, model_path=parse_args.model_path)
    #stats.plot_model_stats(model, pre_train_embeddings, post_train_embeddings, original_model_embeddings)
    #best_k = stats.find_optimal_k(post_train_embeddings, k_range=range(10, 100))
    #stats.cluster_embeddings(post_train_embeddings, best_k)
    stats.compare_clustering_algorithms(post_train_embeddings, 10)
