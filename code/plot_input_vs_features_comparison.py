"""
Plot comparison of input spectrogram and extracted features:
- Left: Raw input (no normalization) → Features
- Right: Normalized input (layer_norm) → Features

Shows how F.layer_norm affects both input distribution and extracted features.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Add fairseq to path
sys.path.insert(0, '/mnt5/noy/SpectralFM/fairseq')
os.environ['USER_DIR'] = '/mnt5/noy/SpectralFM/fairseq/examples/data2vec'

from fairseq.models.wav2vec.wav2vec2 import ConvFeatureExtractionModel
from fairseq.modules import LayerNorm


def load_raw_audio(manifest_path, sample_idx=0):
    """Load raw audio from manifest file."""
    with open(manifest_path, 'r') as f:
        root_dir = f.readline().strip()
        for i, line in enumerate(f):
            if i == sample_idx:
                items = line.strip().split('\t')
                fname = items[0]
                break
    
    audio_path = os.path.join(root_dir, fname)
    wav, sample_rate = sf.read(audio_path, dtype='float32')
    return torch.from_numpy(wav).float(), sample_rate, fname


def plot_input_features_comparison(manifest_path, sample_idx=0, output_dir=None):
    """
    Create a side-by-side comparison plot:
    - Left column: Raw input → Features (normalize=False)
    - Right column: Normalized input → Features (normalize=True)
    """
    if output_dir is None:
        output_dir = '/mnt5/noy/SpectralFM/code/eval_results/normalization_stages'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw audio
    raw_audio, sr, fname = load_raw_audio(manifest_path, sample_idx)
    print(f"Loaded: {fname}, {len(raw_audio)} samples")
    
    if raw_audio.dim() == 2:
        raw_audio = raw_audio.mean(-1)
    
    # Create feature extractor (same as data2vec)
    conv_layers = [(512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 5, 5)]
    feature_extractor = ConvFeatureExtractionModel(
        conv_layers=conv_layers,
        dropout=0.0,
        mode='layer_norm',
        conv_bias=False,
    )
    feature_extractor.eval()
    
    post_cnn_layer_norm = LayerNorm(512)
    
    with torch.no_grad():
        # === Path 1: No normalization (normalize=False) ===
        input_raw = raw_audio.clone()
        features_raw = feature_extractor(input_raw.unsqueeze(0))  # [1, 512, T']
        features_raw_normed = post_cnn_layer_norm(features_raw.transpose(1, 2))  # [1, T', 512]
        
        # === Path 2: With normalization (normalize=True) ===
        input_norm = F.layer_norm(raw_audio, raw_audio.shape)
        features_norm = feature_extractor(input_norm.unsqueeze(0))  # [1, 512, T']
        features_norm_normed = post_cnn_layer_norm(features_norm.transpose(1, 2))  # [1, T', 512]
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    
    # Create grid: 4 rows x 2 columns
    # Row 1: Input waveform
    # Row 2: Input distribution
    # Row 3: Features (after CNN)
    # Row 4: Final features (after post-CNN LayerNorm)
    
    # Colors
    color_raw = '#2E86AB'  # Blue for raw
    color_norm = '#A23B72'  # Purple for normalized
    
    # ========== LEFT COLUMN: normalize=False ==========
    
    # Row 1: Raw input waveform
    ax1 = fig.add_subplot(4, 2, 1)
    data = input_raw.numpy()
    ax1.plot(data, linewidth=0.8, color=color_raw)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=-1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    ax1.fill_between(range(len(data)), data, alpha=0.3, color=color_raw)
    ax1.set_title(f'INPUT: normalize=False (Raw)\nRange: [{data.min():.3f}, {data.max():.3f}], μ={data.mean():.4f}, σ={data.std():.4f}', 
                  fontsize=11, fontweight='bold', color=color_raw)
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.set_xlim(0, len(data))
    ax1.set_ylim(-0.5, 1.5)
    
    # Row 2: Raw input distribution
    ax2 = fig.add_subplot(4, 2, 3)
    ax2.hist(data, bins=50, color=color_raw, alpha=0.7, edgecolor='black', linewidth=0.5, density=True)
    ax2.axvline(x=1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(x=-1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    ax2.set_title('Input Distribution', fontsize=10)
    ax2.set_xlabel('Value', fontsize=9)
    ax2.set_ylabel('Density', fontsize=9)
    
    # Row 3: Features after CNN (raw input)
    ax3 = fig.add_subplot(4, 2, 5)
    feat_data = features_raw[0].numpy()  # [512, T']
    im3 = ax3.imshow(feat_data, aspect='auto', cmap='viridis', interpolation='nearest')
    ax3.set_title(f'CNN Features (from raw input)\nShape: {feat_data.shape}, Range: [{feat_data.min():.2f}, {feat_data.max():.2f}]', fontsize=10)
    ax3.set_ylabel('Channel (512)', fontsize=9)
    ax3.set_xlabel('Time', fontsize=9)
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # Row 4: Final features after post-CNN LayerNorm (raw input)
    ax4 = fig.add_subplot(4, 2, 7)
    final_data = features_raw_normed[0].numpy()  # [T', 512]
    im4 = ax4.imshow(final_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax4.set_title(f'Final Features (after post-CNN LN)\nμ={final_data.mean():.4f}, σ={final_data.std():.4f}, Range: [{final_data.min():.2f}, {final_data.max():.2f}]', fontsize=10)
    ax4.set_ylabel('Channel (512)', fontsize=9)
    ax4.set_xlabel('Time', fontsize=9)
    plt.colorbar(im4, ax=ax4, shrink=0.8)
    
    # ========== RIGHT COLUMN: normalize=True ==========
    
    # Row 1: Normalized input waveform
    ax5 = fig.add_subplot(4, 2, 2)
    data = input_norm.numpy()
    ax5.plot(data, linewidth=0.8, color=color_norm)
    ax5.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax5.axhline(y=-1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    ax5.fill_between(range(len(data)), data, alpha=0.3, color=color_norm)
    ax5.set_title(f'INPUT: normalize=True (F.layer_norm)\nRange: [{data.min():.3f}, {data.max():.3f}], μ={data.mean():.6f}, σ={data.std():.4f}', 
                  fontsize=11, fontweight='bold', color=color_norm)
    ax5.set_xlim(0, len(data))
    ax5.set_ylim(-4, 4)  # Wider range for normalized data
    
    # Row 2: Normalized input distribution
    ax6 = fig.add_subplot(4, 2, 4)
    ax6.hist(data, bins=50, color=color_norm, alpha=0.7, edgecolor='black', linewidth=0.5, density=True)
    ax6.axvline(x=1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax6.axvline(x=-1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax6.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    # Overlay standard normal
    x_range = np.linspace(-4, 4, 100)
    normal_pdf = np.exp(-x_range**2 / 2) / np.sqrt(2 * np.pi)
    ax6.plot(x_range, normal_pdf, 'k--', linewidth=2, alpha=0.7, label='Standard Normal')
    ax6.set_title('Input Distribution (≈ Standard Normal)', fontsize=10)
    ax6.set_xlabel('Value', fontsize=9)
    ax6.legend(fontsize=8)
    
    # Row 3: Features after CNN (normalized input)
    ax7 = fig.add_subplot(4, 2, 6)
    feat_data = features_norm[0].numpy()  # [512, T']
    im7 = ax7.imshow(feat_data, aspect='auto', cmap='viridis', interpolation='nearest')
    ax7.set_title(f'CNN Features (from normalized input)\nShape: {feat_data.shape}, Range: [{feat_data.min():.2f}, {feat_data.max():.2f}]', fontsize=10)
    ax7.set_ylabel('Channel (512)', fontsize=9)
    ax7.set_xlabel('Time', fontsize=9)
    plt.colorbar(im7, ax=ax7, shrink=0.8)
    
    # Row 4: Final features after post-CNN LayerNorm (normalized input)
    ax8 = fig.add_subplot(4, 2, 8)
    final_data = features_norm_normed[0].numpy()  # [T', 512]
    im8 = ax8.imshow(final_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax8.set_title(f'Final Features (after post-CNN LN)\nμ={final_data.mean():.4f}, σ={final_data.std():.4f}, Range: [{final_data.min():.2f}, {final_data.max():.2f}]', fontsize=10)
    ax8.set_ylabel('Channel (512)', fontsize=9)
    ax8.set_xlabel('Time', fontsize=9)
    plt.colorbar(im8, ax=ax8, shrink=0.8)
    
    # Main title
    fig.suptitle(f'Input Spectrogram & Extracted Features Comparison\n'
                 f'File: {fname}\n\n'
                 f'Left: normalize=False (raw input) | Right: normalize=True (F.layer_norm applied)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'input_vs_features_comparison_sample{sample_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY - Sample {sample_idx}")
    print(f"{'='*70}")
    print(f"\n{'normalize=False (Raw)':<35} | {'normalize=True (LayerNorm)':<35}")
    print("-" * 75)
    print(f"Input range:  [{input_raw.min():.3f}, {input_raw.max():.3f}]           | [{input_norm.min():.3f}, {input_norm.max():.3f}]")
    print(f"Input μ:      {input_raw.mean():.4f}                        | {input_norm.mean():.6f}")
    print(f"Input σ:      {input_raw.std():.4f}                        | {input_norm.std():.4f}")
    print(f"CNN out range:[{features_raw.min():.2f}, {features_raw.max():.2f}]               | [{features_norm.min():.2f}, {features_norm.max():.2f}]")
    print(f"Final μ:      {features_raw_normed.mean():.6f}                   | {features_norm_normed.mean():.6f}")
    print(f"Final σ:      {features_raw_normed.std():.4f}                       | {features_norm_normed.std():.4f}")
    print(f"\nPlot saved to: {save_path}")
    
    return save_path


if __name__ == '__main__':
    manifest_path = '/mnt5/noy/fairseq/data/single_channel_10k/train.tsv'
    
    # Generate plots for several samples
    for sample_idx in [0, 50, 100, 200]:
        try:
            plot_input_features_comparison(manifest_path, sample_idx=sample_idx)
        except Exception as e:
            print(f"Error on sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
