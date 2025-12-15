import soundfile as sf
import numpy as np

# Load one of your new files
data, rate = sf.read('data/single_channel_1m/spec_0.wav')

print(f"Shape: {data.shape}")  # Should be (245,)
print(f"Type:  {data.dtype}")  # Should be float32
print(f"Min:   {data.min()}")  # Should NOT be -1.0 (unless your data is normalized)
print(f"Max:   {data.max()}")  # Should NOT be 1.0

# If Max > 1.0 or Min < -1.0, CONGRATS! 
# It means you successfully saved the spectrogram values without clipping.
