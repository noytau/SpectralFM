"""
Side-by-side model comparison visualization.
Compares outputs from:
1. Input data (original)
2. Pretrained base_libri model
3. Feature extractor only trained model
4. Full trained model (best checkpoint)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import glob
import os
import random
import sys

# Add fairseq to path
_FAIRSEQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fairseq")
if _FAIRSEQ_PATH not in sys.path:
    sys.path.insert(0, _FAIRSEQ_PATH)

from model_loader import load_fairseq_checkpoint

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# ============== MODEL PATHS ==============
# Pretrained base model
BASE_LIBRI_PATH = "/mnt5/noy/SpectralFM/fairseq/base_libri.pt"

# FE-only trained models (loss > 6, trained before train_only_fe param was added)
FE_ONLY_CHECKPOINTS = {
    "2025-12-27_00-06-01": {
        "path": "/mnt5/noy/SpectralFM/checkpoints/runai/2025-12-27_00-06-01/checkpoint_best.pt",
        "val_loss": 6.191,
        "train_only_fe": True,  # Marked as FE-only based on loss > 6
    },
    "2025-12-27_07-26-47": {
        "path": "/mnt5/noy/SpectralFM/checkpoints/runai/2025-12-27_07-26-47/checkpoint_best.pt",
        "val_loss": 6.324,
        "train_only_fe": True,  # Marked as FE-only based on loss > 6
    },
}

# Full train checkpoints (loss < 6)
FULL_TRAIN_CHECKPOINTS = {
    "2025-12-27_09-59-50": {
        "path": "/mnt5/noy/SpectralFM/checkpoints/runai/2025-12-27_09-59-50/checkpoint_best.pt",
        "val_loss": 0.729,  # BEST - lowest loss
        "train_only_fe": False,
    },
    "2025-12-27_22-00-53": {
        "path": "/mnt5/noy/SpectralFM/checkpoints/runai/2025-12-27_22-00-53/checkpoint_best.pt",
        "val_loss": 1.016,
        "train_only_fe": False,
    },
    "2025-12-31_12-53-44": {
        "path": "/mnt5/noy/SpectralFM/checkpoints/runai/2025-12-31_12-53-44/checkpoint_best.pt",
        "val_loss": 1.439,
        "train_only_fe": False,
    },
    "2026-01-07_12-26-41": {
        "path": "/mnt5/noy/SpectralFM/checkpoints/runai/2026-01-07_12-26-41/checkpoint_best.pt",
        "val_loss": 1.227,
        "train_only_fe": False,
    },
}

# Default selections
DEFAULT_FE_ONLY = "2025-12-27_00-06-01"  # Pick first FE-only run
DEFAULT_FULL_TRAIN = "2025-12-27_09-59-50"  # Best full train (lowest loss)


def load_sample_signals(data_dir, n_samples=3, seed=42):
    """Load sample wav files for evaluation."""
    random.seed(seed)
    wav_files = glob.glob(os.path.join(data_dir, "*.wav"))
    if not wav_files:
        raise ValueError(f"No wav files found in {data_dir}")
    
    sampled_files = random.sample(wav_files, min(n_samples, len(wav_files)))
    
    signals = []
    for wav_path in sampled_files:
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        signals.append({
            'waveform': waveform,
            'filename': os.path.basename(wav_path),
            'sr': 16000
        })
    
    return signals


def load_all_models(device=None):
    """Load all models for comparison."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = {}
    
    # 1. Load pretrained base_libri model
    print("\n[1/3] Loading pretrained base_libri model...")
    try:
        model, cfg, info = load_fairseq_checkpoint(BASE_LIBRI_PATH, device)
        models['pretrained'] = {
            'model': model,
            'cfg': cfg,
            'info': info,
            'name': 'Pretrained (base_libri)',
            'color': '#3498DB'
        }
    except Exception as e:
        print(f"[!] Failed to load pretrained model: {e}")
        models['pretrained'] = None
    
    # 2. Load FE-only trained model
    print("\n[2/3] Loading FE-only trained model...")
    fe_info = FE_ONLY_CHECKPOINTS[DEFAULT_FE_ONLY]
    try:
        model, cfg, info = load_fairseq_checkpoint(fe_info['path'], device)
        models['fe_only'] = {
            'model': model,
            'cfg': cfg,
            'info': info,
            'name': f'FE-Only ({DEFAULT_FE_ONLY})\nLoss: {fe_info["val_loss"]:.3f}',
            'color': '#E74C3C'
        }
    except Exception as e:
        print(f"[!] Failed to load FE-only model: {e}")
        models['fe_only'] = None
    
    # 3. Load full train model (best)
    print("\n[3/3] Loading full train model (best)...")
    full_info = FULL_TRAIN_CHECKPOINTS[DEFAULT_FULL_TRAIN]
    try:
        model, cfg, info = load_fairseq_checkpoint(full_info['path'], device)
        models['full_train'] = {
            'model': model,
            'cfg': cfg,
            'info': info,
            'name': f'Full Train ({DEFAULT_FULL_TRAIN})\nLoss: {full_info["val_loss"]:.3f}',
            'color': '#27AE60'
        }
    except Exception as e:
        print(f"[!] Failed to load full train model: {e}")
        models['full_train'] = None
    
    return models, device


def extract_features(model, waveform, device):
    """Extract features/embeddings from a model given a waveform."""
    model.eval()
    with torch.no_grad():
        # Ensure waveform is on the correct device
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add batch dimension
        waveform = waveform.to(device)
        
        # Get model output
        try:
            output = model(waveform, features_only=True)
            if hasattr(output, 'x'):
                features = output.x
            elif isinstance(output, dict) and 'x' in output:
                features = output['x']
            elif isinstance(output, tuple):
                features = output[0]
            else:
                features = output
            return features.cpu()
        except Exception as e:
            print(f"[!] Feature extraction failed: {e}")
            return None


def compute_spectrogram(waveform, n_fft=512, hop_length=160, n_mels=80):
    """Compute mel spectrogram from waveform."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.squeeze().numpy()
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    waveform_tensor = torch.tensor(waveform).float()
    if waveform_tensor.dim() == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)
    
    mel_spec = mel_transform(waveform_tensor)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    return mel_spec_db.squeeze().numpy()


def plot_side_by_side_comparison(signals, models, output_dir, device):
    """Create side-by-side comparison plots for all signals."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Model keys in order
    model_keys = ['pretrained', 'fe_only', 'full_train']
    
    for sig_idx, signal_data in enumerate(signals):
        waveform = signal_data['waveform']
        filename = signal_data['filename']
        
        # Create figure with 4 columns (input + 3 models) and 3 rows
        # Row 1: Waveform
        # Row 2: Spectrogram  
        # Row 3: Feature embeddings (mean over time)
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        fig.suptitle(f'Model Comparison: {filename}', fontsize=16, fontweight='bold', y=1.02)
        
        # Column 0: Input data
        time = np.arange(waveform.shape[-1]) / 16000
        waveform_np = waveform.squeeze().numpy()
        
        # Row 0, Col 0: Input waveform
        axes[0, 0].plot(time, waveform_np, color='#2C3E50', linewidth=0.5)
        axes[0, 0].set_title('Input Signal\n(Original)', fontsize=12, fontweight='bold', color='#2C3E50')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_xlim(0, time[-1])
        
        # Row 1, Col 0: Input spectrogram
        input_spec = compute_spectrogram(waveform)
        im0 = axes[1, 0].imshow(input_spec, aspect='auto', origin='lower', cmap='magma')
        axes[1, 0].set_title('Input Spectrogram', fontsize=11)
        axes[1, 0].set_xlabel('Time frames')
        axes[1, 0].set_ylabel('Mel bins')
        plt.colorbar(im0, ax=axes[1, 0], format='%.0f dB')
        
        # Row 2, Col 0: Placeholder for input (no embeddings)
        axes[2, 0].text(0.5, 0.5, 'N/A\n(Raw Input)', ha='center', va='center', 
                        fontsize=14, color='#7F8C8D', transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('Embeddings', fontsize=11)
        axes[2, 0].axis('off')
        
        # Columns 1-3: Model outputs
        for col_idx, model_key in enumerate(model_keys, start=1):
            model_data = models.get(model_key)
            
            if model_data is None:
                # Model not loaded - show placeholder
                for row_idx in range(3):
                    axes[row_idx, col_idx].text(0.5, 0.5, 'Model\nNot Loaded', 
                                                 ha='center', va='center', fontsize=12,
                                                 color='#E74C3C', transform=axes[row_idx, col_idx].transAxes)
                    axes[row_idx, col_idx].axis('off')
                continue
            
            model = model_data['model']
            model_name = model_data['name']
            model_color = model_data['color']
            
            # Extract features
            features = extract_features(model, waveform, device)
            
            # Row 0: Waveform (same as input, for reference)
            axes[0, col_idx].plot(time, waveform_np, color=model_color, linewidth=0.5, alpha=0.7)
            axes[0, col_idx].set_title(f'{model_name}', fontsize=12, fontweight='bold', color=model_color)
            axes[0, col_idx].set_xlabel('Time (s)')
            axes[0, col_idx].set_ylabel('Amplitude')
            axes[0, col_idx].set_xlim(0, time[-1])
            
            # Row 1: Spectrogram (same input, different processing context)
            axes[1, col_idx].imshow(input_spec, aspect='auto', origin='lower', cmap='magma')
            axes[1, col_idx].set_title('Input Spectrogram', fontsize=11)
            axes[1, col_idx].set_xlabel('Time frames')
            axes[1, col_idx].set_ylabel('Mel bins')
            
            # Row 2: Feature embeddings
            if features is not None:
                features_np = features.squeeze().numpy()
                if features_np.ndim == 2:
                    # Show feature heatmap (time x features)
                    im = axes[2, col_idx].imshow(features_np.T, aspect='auto', origin='lower', cmap='viridis')
                    axes[2, col_idx].set_title(f'Features: {features_np.shape}', fontsize=11)
                    axes[2, col_idx].set_xlabel('Time frames')
                    axes[2, col_idx].set_ylabel('Feature dim')
                    plt.colorbar(im, ax=axes[2, col_idx])
                elif features_np.ndim == 1:
                    # 1D features - plot as bar
                    axes[2, col_idx].bar(range(len(features_np)), features_np, color=model_color, alpha=0.7)
                    axes[2, col_idx].set_title(f'Features: {features_np.shape}', fontsize=11)
                    axes[2, col_idx].set_xlabel('Feature index')
                    axes[2, col_idx].set_ylabel('Value')
            else:
                axes[2, col_idx].text(0.5, 0.5, 'Feature extraction\nfailed', 
                                       ha='center', va='center', fontsize=12,
                                       color='#E74C3C', transform=axes[2, col_idx].transAxes)
                axes[2, col_idx].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'model_comparison_signal_{sig_idx + 1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[+] Saved: {save_path}")
    
    # Create summary info
    create_model_summary(models, output_dir)


def plot_embedding_similarity_comparison(signals, models, output_dir, device):
    """Compare embedding similarities across models."""
    
    os.makedirs(output_dir, exist_ok=True)
    model_keys = ['pretrained', 'fe_only', 'full_train']
    
    # Collect all embeddings
    all_embeddings = {key: [] for key in model_keys}
    
    for signal_data in signals:
        waveform = signal_data['waveform']
        
        for model_key in model_keys:
            model_data = models.get(model_key)
            if model_data is None:
                all_embeddings[model_key].append(None)
                continue
            
            features = extract_features(model_data['model'], waveform, device)
            if features is not None:
                # Mean pool over time to get single embedding
                emb = features.squeeze().mean(dim=0).numpy()
                all_embeddings[model_key].append(emb)
            else:
                all_embeddings[model_key].append(None)
    
    # Compute pairwise cosine similarities for each model
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Embedding Similarity Matrices Across Models', fontsize=14, fontweight='bold')
    
    for idx, model_key in enumerate(model_keys):
        embeddings = all_embeddings[model_key]
        valid_embs = [e for e in embeddings if e is not None]
        
        if len(valid_embs) < 2:
            axes[idx].text(0.5, 0.5, 'Not enough\nvalid embeddings', 
                          ha='center', va='center', fontsize=12)
            axes[idx].axis('off')
            continue
        
        # Stack and compute similarity
        emb_matrix = np.stack(valid_embs)
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(emb_matrix)
        
        model_data = models.get(model_key)
        model_name = model_data['name'].split('\n')[0] if model_data else model_key
        
        im = axes[idx].imshow(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Sample')
        axes[idx].set_ylabel('Sample')
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'embedding_similarity_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_similarity_matrices(data_dir, models, output_dir, device, n_samples=50):
    """
    Generate side-by-side similarity matrix comparison:
    Input vs Pretrained vs FE-Only vs Full Train embeddings.
    
    Adapted from evaluation_runner.plot_similarity_matrices.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm import tqdm
    import seaborn as sns
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load wav files
    wav_files = glob.glob(os.path.join(data_dir, "*.wav"))[:n_samples]
    print(f"[+] Loading {len(wav_files)} samples for similarity matrix computation...")
    
    # Collect inputs and embeddings from all models
    inputs = []
    model_keys = ['pretrained', 'fe_only', 'full_train']
    all_embeddings = {key: [] for key in model_keys}
    
    with torch.no_grad():
        for wav_path in tqdm(wav_files, desc="Extracting embeddings"):
            try:
                waveform, sr = torchaudio.load(wav_path)
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Store raw input
                inputs.append(waveform.squeeze(0).cpu().numpy())
                
                # Extract embeddings from each model
                for model_key in model_keys:
                    model_data = models.get(model_key)
                    if model_data is None:
                        all_embeddings[model_key].append(None)
                        continue
                    
                    model = model_data['model']
                    model.eval()
                    
                    data = waveform.squeeze(0).unsqueeze(0).to(device)
                    try:
                        result = model.extract_features(data, padding_mask=None, mask=False)
                        emb = result["x"].mean(dim=1).cpu().numpy().squeeze()
                        all_embeddings[model_key].append(emb)
                    except Exception as e:
                        # Fallback: try features_only mode
                        try:
                            output = model(data, features_only=True)
                            if hasattr(output, 'x'):
                                emb = output.x.mean(dim=1).cpu().numpy().squeeze()
                            elif isinstance(output, dict) and 'x' in output:
                                emb = output['x'].mean(dim=1).cpu().numpy().squeeze()
                            else:
                                emb = output[0].mean(dim=1).cpu().numpy().squeeze() if isinstance(output, tuple) else None
                            all_embeddings[model_key].append(emb)
                        except:
                            all_embeddings[model_key].append(None)
                            
            except Exception as e:
                print(f"[!] Error processing {wav_path}: {e}")
                continue
    
    if len(inputs) < 2:
        print("[!] Not enough samples loaded for similarity matrix")
        return {}
    
    # Stack inputs
    inputs = np.stack(inputs)
    
    # Compute input similarity matrix
    input_sim = cosine_similarity(inputs)
    
    # Compute embedding similarity matrices for each model
    emb_sim_matrices = {}
    for model_key in model_keys:
        embeddings = all_embeddings[model_key]
        valid_embs = [e for e in embeddings if e is not None]
        if len(valid_embs) >= 2:
            emb_matrix = np.stack(valid_embs)
            emb_sim_matrices[model_key] = cosine_similarity(emb_matrix)
        else:
            emb_sim_matrices[model_key] = None
    
    # ============== PLOT 1: 4-way comparison (2x2 grid) ==============
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Cosine Similarity Matrix Comparison\nInput vs Pretrained vs FE-Only vs Full Train', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Define plot configs
    plot_configs = [
        (0, 0, 'Input Space', input_sim, '#2C3E50'),
        (0, 1, 'Pretrained (base_libri)', emb_sim_matrices.get('pretrained'), '#3498DB'),
        (1, 0, f'FE-Only (loss: {FE_ONLY_CHECKPOINTS[DEFAULT_FE_ONLY]["val_loss"]:.3f})', 
         emb_sim_matrices.get('fe_only'), '#E74C3C'),
        (1, 1, f'Full Train (loss: {FULL_TRAIN_CHECKPOINTS[DEFAULT_FULL_TRAIN]["val_loss"]:.3f})', 
         emb_sim_matrices.get('full_train'), '#27AE60'),
    ]
    
    for row, col, title, sim_matrix, color in plot_configs:
        ax = axes[row, col]
        if sim_matrix is not None:
            sns.heatmap(sim_matrix, ax=ax, cmap='viridis', 
                       xticklabels=False, yticklabels=False,
                       vmin=0, vmax=1, cbar_kws={'label': 'Cosine Similarity'})
            ax.set_title(title, fontsize=12, fontweight='bold', color=color)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Sample Index')
            
            # Add statistics
            mean_sim = sim_matrix[np.triu_indices_from(sim_matrix, k=1)].mean()
            std_sim = sim_matrix[np.triu_indices_from(sim_matrix, k=1)].std()
            ax.text(0.02, 0.98, f'Mean: {mean_sim:.3f}\nStd: {std_sim:.3f}', 
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Model Not Loaded\nor Extraction Failed', 
                   ha='center', va='center', fontsize=12, color='#E74C3C')
            ax.set_title(title, fontsize=12, fontweight='bold', color=color)
            ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'similarity_matrices_4way_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved 4-way comparison: {save_path}")
    
    # ============== PLOT 2: Difference matrices (relative to input) ==============
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Embedding Similarity - Input Similarity (Difference Matrices)', 
                 fontsize=14, fontweight='bold')
    
    for idx, model_key in enumerate(model_keys):
        ax = axes[idx]
        model_data = models.get(model_key)
        emb_sim = emb_sim_matrices.get(model_key)
        
        if emb_sim is not None and emb_sim.shape == input_sim.shape:
            diff_matrix = emb_sim - input_sim
            
            model_name = model_data['name'].split('\n')[0] if model_data else model_key
            color = model_data['color'] if model_data else '#7F8C8D'
            
            sns.heatmap(diff_matrix, ax=ax, cmap='RdBu_r', center=0,
                       xticklabels=False, yticklabels=False,
                       vmin=-1, vmax=1, cbar_kws={'label': 'Difference'})
            ax.set_title(f'{model_name}\n(Embedding - Input)', fontsize=12, fontweight='bold', color=color)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Sample Index')
            
            # Add statistics
            mean_diff = diff_matrix[np.triu_indices_from(diff_matrix, k=1)].mean()
            ax.text(0.02, 0.98, f'Mean Diff: {mean_diff:+.3f}', 
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
            ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'similarity_matrices_diff_from_input.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved difference matrices: {save_path}")
    
    # ============== PLOT 3: Histogram comparison ==============
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Distribution of Pairwise Cosine Similarities', fontsize=14, fontweight='bold')
    
    # Get upper triangular values (excluding diagonal)
    input_vals = input_sim[np.triu_indices_from(input_sim, k=1)]
    ax.hist(input_vals, bins=50, alpha=0.6, label='Input Space', color='#2C3E50', density=True)
    
    colors = {'pretrained': '#3498DB', 'fe_only': '#E74C3C', 'full_train': '#27AE60'}
    labels = {
        'pretrained': 'Pretrained (base_libri)',
        'fe_only': f'FE-Only (loss: {FE_ONLY_CHECKPOINTS[DEFAULT_FE_ONLY]["val_loss"]:.3f})',
        'full_train': f'Full Train (loss: {FULL_TRAIN_CHECKPOINTS[DEFAULT_FULL_TRAIN]["val_loss"]:.3f})'
    }
    
    for model_key in model_keys:
        emb_sim = emb_sim_matrices.get(model_key)
        if emb_sim is not None:
            emb_vals = emb_sim[np.triu_indices_from(emb_sim, k=1)]
            ax.hist(emb_vals, bins=50, alpha=0.5, label=labels[model_key], 
                   color=colors[model_key], density=True)
    
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(-0.2, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'similarity_distribution_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved distribution comparison: {save_path}")
    
    # ============== PLOT 4: Correlation between models ==============
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Correlation: Input Similarity vs Embedding Similarity', 
                 fontsize=14, fontweight='bold')
    
    input_vals = input_sim[np.triu_indices_from(input_sim, k=1)]
    
    for idx, model_key in enumerate(model_keys):
        ax = axes[idx]
        emb_sim = emb_sim_matrices.get(model_key)
        model_data = models.get(model_key)
        
        if emb_sim is not None and emb_sim.shape == input_sim.shape:
            emb_vals = emb_sim[np.triu_indices_from(emb_sim, k=1)]
            
            model_name = model_data['name'].split('\n')[0] if model_data else model_key
            color = model_data['color'] if model_data else '#7F8C8D'
            
            ax.scatter(input_vals, emb_vals, alpha=0.3, s=5, color=color)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
            
            # Compute correlation
            correlation = np.corrcoef(input_vals, emb_vals)[0, 1]
            
            ax.set_title(f'{model_name}\nCorrelation: {correlation:.3f}', 
                        fontsize=12, fontweight='bold', color=color)
            ax.set_xlabel('Input Similarity')
            ax.set_ylabel('Embedding Similarity')
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
            ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'input_vs_embedding_correlation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved correlation plots: {save_path}")
    
    # Return data for further analysis
    return {
        "input_sim_matrix": input_sim,
        "pretrained_sim_matrix": emb_sim_matrices.get('pretrained'),
        "fe_only_sim_matrix": emb_sim_matrices.get('fe_only'),
        "full_train_sim_matrix": emb_sim_matrices.get('full_train'),
        "n_samples": len(inputs),
    }


def create_model_summary(models, output_dir):
    """Create a text summary of loaded models."""
    
    summary_path = os.path.join(output_dir, 'model_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("FE-ONLY TRAINED MODELS (loss > 6):\n")
        f.write("-" * 40 + "\n")
        for name, info in FE_ONLY_CHECKPOINTS.items():
            f.write(f"  {name}:\n")
            f.write(f"    Path: {info['path']}\n")
            f.write(f"    Val Loss: {info['val_loss']}\n")
            f.write(f"    train_only_fe: {info['train_only_fe']}\n\n")
        
        f.write("\nFULL TRAIN MODELS (loss < 6):\n")
        f.write("-" * 40 + "\n")
        for name, info in sorted(FULL_TRAIN_CHECKPOINTS.items(), key=lambda x: x[1]['val_loss']):
            marker = " *** BEST ***" if name == DEFAULT_FULL_TRAIN else ""
            f.write(f"  {name}:{marker}\n")
            f.write(f"    Path: {info['path']}\n")
            f.write(f"    Val Loss: {info['val_loss']}\n")
            f.write(f"    train_only_fe: {info['train_only_fe']}\n\n")
        
        f.write("\nSELECTED FOR COMPARISON:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Pretrained: {BASE_LIBRI_PATH}\n")
        f.write(f"  FE-Only: {DEFAULT_FE_ONLY}\n")
        f.write(f"  Full Train: {DEFAULT_FULL_TRAIN}\n")
    
    print(f"[+] Saved summary: {summary_path}")


def main():
    # Configuration
    data_dir = "/mnt5/noy/fairseq/data/single_channel_1m/"
    output_dir = "/mnt5/noy/SpectralFM/code/eval_results/model_comparison"
    n_samples_visual = 5  # For per-signal visualizations
    n_samples_similarity = 50  # For similarity matrix computation
    
    print("=" * 60)
    print("MODEL COMPARISON EVALUATION")
    print("=" * 60)
    
    # Load models
    print("\n[+] Loading models...")
    models, device = load_all_models()
    
    # Load sample signals for visual comparison
    print(f"\n[+] Loading {n_samples_visual} sample signals for visual comparison...")
    signals = load_sample_signals(data_dir, n_samples=n_samples_visual)
    print(f"    Loaded {len(signals)} signals")
    
    # Create side-by-side comparison
    print("\n[+] Creating side-by-side comparison plots...")
    plot_side_by_side_comparison(signals, models, output_dir, device)
    
    # Create embedding similarity comparison (small scale)
    print("\n[+] Creating embedding similarity comparison...")
    plot_embedding_similarity_comparison(signals, models, output_dir, device)
    
    # Create comprehensive similarity matrices (larger scale)
    print(f"\n[+] Computing similarity matrices on {n_samples_similarity} samples...")
    print("    This compares: Input vs Pretrained vs FE-Only vs Full Train")
    sim_results = plot_similarity_matrices(data_dir, models, output_dir, device, 
                                            n_samples=n_samples_similarity)
    
    # Print summary statistics
    if sim_results:
        print("\n" + "=" * 60)
        print("SIMILARITY MATRIX SUMMARY")
        print("=" * 60)
        print(f"  Samples analyzed: {sim_results['n_samples']}")
        
        for key in ['input_sim_matrix', 'pretrained_sim_matrix', 'fe_only_sim_matrix', 'full_train_sim_matrix']:
            matrix = sim_results.get(key)
            if matrix is not None:
                upper_vals = matrix[np.triu_indices_from(matrix, k=1)]
                name = key.replace('_sim_matrix', '').replace('_', ' ').title()
                print(f"  {name}:")
                print(f"    Mean similarity: {upper_vals.mean():.4f}")
                print(f"    Std similarity:  {upper_vals.std():.4f}")
    
    print(f"\n[+] Done! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
