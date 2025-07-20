import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compute_cosine_similarity_matrix(dataset, output_dir="plots", name="spectogram"):
    # Get all data from the specified column
    data_tensor = torch.tensor(dataset, dtype=torch.float32)  # shape: [N, 16000]

    # Normalize for cosine similarity
    normalized = torch.nn.functional.normalize(data_tensor, p=2, dim=1)

    # Compute cosine similarity matrix
    sim_matrix = torch.matmul(normalized, normalized.T).cpu().numpy()

    # Plot heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, cmap="viridis", xticklabels=False, yticklabels=False)
    plt.title(f"Cosine Similarity Heatmap ({name})")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    #plt.show()
    #plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"cosine_similarity_{name}.png"))
    plt.close()

    return sim_matrix

def compute_cosine_similarity_matrix_from_embeddings(embeddings, output_dir="plots"):
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
    plt.savefig(os.path.join(output_dir, "cosine_similarity_embeddings.png"))
    plt.close()

    return sim_matrix