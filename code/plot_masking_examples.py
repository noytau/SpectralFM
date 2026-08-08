"""
Visualize the difference between causal and random masking strategies.
Creates plots showing original signals with masked regions highlighted.
"""

import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import torch
import glob
import os
import random

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def load_sample_signals(data_dir, n_samples=2, seed=42):
    """Load sample wav files."""
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
            'waveform': waveform.squeeze().numpy(),
            'filename': os.path.basename(wav_path)
        })
    
    return signals


def create_causal_mask(length, mask_ratio):
    """Create causal mask - masks the END of the signal."""
    mask = np.zeros(length, dtype=bool)
    mask_start = int(length * (1 - mask_ratio))
    mask[mask_start:] = True
    return mask


def create_random_mask(length, mask_ratio, seed=None):
    """Create random mask - masks random positions throughout."""
    if seed is not None:
        np.random.seed(seed)
    mask = np.zeros(length, dtype=bool)
    num_masked = int(length * mask_ratio)
    masked_indices = np.random.choice(length, num_masked, replace=False)
    mask[masked_indices] = True
    return mask


def plot_masking_comparison(signals, output_dir):
    """Create comparison plots for causal vs random masking."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define masking strategies
    strategies = {
        'causal_50': {'type': 'causal', 'ratio': 0.5, 'color': '#E74C3C', 'label': 'Causal 50%\n(mask last half)'},
        'causal_25': {'type': 'causal', 'ratio': 0.25, 'color': '#E67E22', 'label': 'Causal 25%\n(mask last quarter)'},
        'random_30': {'type': 'random', 'ratio': 0.3, 'color': '#3498DB', 'label': 'Random 30%\n(random positions)'},
        'random_50': {'type': 'random', 'ratio': 0.5, 'color': '#9B59B6', 'label': 'Random 50%\n(random positions)'},
    }
    
    for sig_idx, signal_data in enumerate(signals):
        waveform = signal_data['waveform']
        filename = signal_data['filename']
        length = len(waveform)
        
        # Create figure with 4 subplots (one per strategy)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Masking Strategies Comparison\nSignal: {filename} (length: {length:,} samples)', 
                     fontsize=14, fontweight='bold', y=1.02)
        
        for idx, (strategy_name, config) in enumerate(strategies.items()):
            ax = axes[idx // 2, idx % 2]
            
            # Create mask
            if config['type'] == 'causal':
                mask = create_causal_mask(length, config['ratio'])
            else:
                mask = create_random_mask(length, config['ratio'], seed=42 + sig_idx)
            
            # Time axis (in seconds assuming 16kHz)
            time = np.arange(length) / 16000
            
            # Plot original signal
            ax.plot(time, waveform, color='#2C3E50', linewidth=0.5, alpha=0.8, label='Original signal')
            
            # Highlight masked regions
            masked_signal = waveform.copy()
            masked_signal[~mask] = np.nan
            ax.plot(time, masked_signal, color=config['color'], linewidth=1.5, alpha=0.9, label='Masked region')
            
            # Add shaded regions for masked areas
            if config['type'] == 'causal':
                # Single shaded region for causal
                mask_start_time = time[np.where(mask)[0][0]]
                ax.axvspan(mask_start_time, time[-1], alpha=0.15, color=config['color'])
                ax.axvline(x=mask_start_time, color=config['color'], linestyle='--', linewidth=2, alpha=0.7)
            
            # Labels and styling
            ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.set_title(f'{strategy_name.upper()}\n{config["label"]}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xlim(0, time[-1])
            
            # Add mask info
            num_masked = mask.sum()
            pct_masked = 100 * num_masked / length
            ax.text(0.02, 0.98, f'Masked: {num_masked:,} samples ({pct_masked:.1f}%)', 
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'masking_comparison_signal_{sig_idx + 1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[+] Saved: {save_path}")
    
    # Create a single explanatory diagram
    create_explanation_diagram(output_dir)


def create_explanation_diagram(output_dir):
    """Create a simple diagram explaining the masking strategies."""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    fig.suptitle('Masking Strategies Explained', fontsize=16, fontweight='bold', y=1.02)
    
    # Create a clean synthetic signal for clarity (no noise added)
    length = 1000
    t = np.linspace(0, 1, length)
    # Clean signal: combination of sine waves (like spectral harmonics)
    signal = np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 7 * t) + 0.3 * np.sin(2 * np.pi * 12 * t)
    
    # Causal masking (top)
    ax = axes[0]
    ax.set_title('CAUSAL MASKING: Mask the END of the signal', fontsize=12, fontweight='bold', color='#C0392B')
    
    causal_mask = create_causal_mask(length, 0.5)
    ax.plot(t, signal, color='#2C3E50', linewidth=1.5, label='Visible (context)')
    masked_sig = signal.copy()
    masked_sig[~causal_mask] = np.nan
    ax.fill_between(t, signal.min() - 0.3, signal.max() + 0.3, where=causal_mask, 
                    alpha=0.3, color='#E74C3C', label='Masked (to predict)')
    ax.plot(t[causal_mask], signal[causal_mask], 'o', color='#E74C3C', markersize=2, alpha=0.5)
    
    ax.axvline(x=0.5, color='#C0392B', linestyle='--', linewidth=2)
    ax.annotate('Mask starts here', xy=(0.5, signal.max()), xytext=(0.3, signal.max() + 0.5),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='#C0392B'),
                color='#C0392B', fontweight='bold')
    ax.annotate('Model sees this\n(past context)', xy=(0.25, 0), fontsize=11, 
                ha='center', color='#2C3E50', fontweight='bold')
    ax.annotate('Model predicts this\n(future)', xy=(0.75, 0), fontsize=11, 
                ha='center', color='#E74C3C', fontweight='bold')
    
    ax.set_xlabel('Time →', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1)
    
    # Random masking (bottom)
    ax = axes[1]
    ax.set_title('RANDOM MASKING: Mask RANDOM positions throughout', fontsize=12, fontweight='bold', color='#2980B9')
    
    random_mask = create_random_mask(length, 0.3, seed=42)
    ax.plot(t, signal, color='#2C3E50', linewidth=1.5, label='Visible (context)')
    
    # Show masked points
    ax.plot(t[random_mask], signal[random_mask], 'o', color='#3498DB', markersize=4, 
            alpha=0.7, label='Masked (to predict)')
    
    ax.annotate('Model uses BOTH\npast and future\nto fill gaps', xy=(0.5, signal.min() - 0.5), 
                fontsize=11, ha='center', color='#2980B9', fontweight='bold')
    
    # Add arrows showing bidirectional context
    ax.annotate('', xy=(0.35, signal.min() - 0.8), xytext=(0.5, signal.min() - 0.8),
                arrowprops=dict(arrowstyle='<->', color='#2980B9', lw=2))
    ax.annotate('', xy=(0.65, signal.min() - 0.8), xytext=(0.5, signal.min() - 0.8),
                arrowprops=dict(arrowstyle='<->', color='#2980B9', lw=2))
    
    ax.set_xlabel('Time →', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'masking_strategies_explained.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved explanation diagram: {save_path}")


if __name__ == "__main__":
    # Configuration
    data_dir = "/mnt5/noy/fairseq/data/single_channel_1m/"
    output_dir = "/mnt5/noy/SpectralFM/code/eval_results/runai_comparison/plots/masking_examples"
    
    print("[+] Loading sample signals...")
    signals = load_sample_signals(data_dir, n_samples=2)
    
    print(f"[+] Creating masking comparison plots for {len(signals)} signals...")
    plot_masking_comparison(signals, output_dir)
    
    print(f"\n[+] Done! Plots saved to: {output_dir}")
