import subprocess
from itertools import product

# Define the options
epochs = [1, 2]
batch_sizes = [16, 20]
mask_ratios = [0.20, 0.30]

# Generate all combinations
combinations = list(product(epochs, batch_sizes, mask_ratios))

# Loop over each combination and run the script
for epoch, batch_size, mask_ratio in combinations:
    cmd = [
        "/home/noy/.virtualenvs/SpectralFM/bin/python", "/mnt5/noy/code/main.py",  # or your script name
        "--test_dir=medium",  # run on 1m samples (1 pkl file)
        f"--epoch={epoch}",
        f"--batch_size={batch_size}",
        f"--mask_ratio={mask_ratio}"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)