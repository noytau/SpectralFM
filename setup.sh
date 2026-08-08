# 1. Manually add conda to your shell session
source /mnt5/noy/miniconda3/etc/profile.d/conda.sh

# 2. Delete the broken environment (if it exists)
conda activate base
conda env remove -n spectralfm_env --yes

# 3. Purge all caches to ensure "broken" metadata isn't saved
conda clean --all -y
pip cache purge

# 1. Create the foundation (Conda packages only)
conda env create -f spectralfm.yml -n spectralfm_env

# 2. Activate using the "Source" method (Robust)
source /mnt5/noy/miniconda3/etc/profile.d/conda.sh
conda activate spectralfm_env

# 3. Apply the "Pip Shield" (Downgrade to a version that ignores bad metadata)
python -m pip install "pip==24.0"

# 4. Force-pin the Golden Trio (Pre-empts Fairseq from pulling the bad 2.0.6)
pip install "omegaconf==2.0.5" "hydra-core==1.0.7" "PyYAML>=5.1,<6.1"

# 5. Finally, install Fairseq as an editable package
cd /mnt5/noy/SpectralFM/fairseq
pip install -e .

git config --global user.email "noyhassid@mail.tau.ac.il"
