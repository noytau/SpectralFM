# 1. Start from your existing lean image
FROM noyhassid/spectralfm-lean:v1

# 2. Install System Compilers (Fixes the g++ error)
RUN apt-get update && \
    apt-get install -y build-essential libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# pin pip < 24.1 to handle omegaconf's metadata issues
RUN python -m pip install --no-cache-dir "pip<24.1"

# 3. Create the Conda Environment (As requested)
COPY spectralfm.yml /tmp/spectralfm.yml
RUN conda env create -f /tmp/spectralfm.yml && \
    conda clean --all -y

# 4. Set the new environment as Default
# This ensures 'python' runs inside your new env automatically
ENV PATH /opt/conda/envs/spectralfm/bin:$PATH
ENV CONDA_DEFAULT_ENV spectralfm

# 5. Downgrade Pip (Safety fix for metadata errors)
RUN pip install "pip==24.0"
RUN pip install "omegaconf==2.0.5" "hydra-core==1.0.7" "PyYAML>=5.1,<6.1"

# Create entrypoint script that sets up symlinks at runtime
# This allows scripts using /mnt5/noy/ paths to resolve on RunAI (mounted at /storage/noy/)
RUN echo '#!/bin/bash\n\
# Create symlink so /mnt5/noy/ resolves to /storage/noy/ at runtime\n\
mkdir -p /mnt5\n\
ln -sfn /storage/noy /mnt5/noy 2>/dev/null || true\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# 6. Bake the Code (Fixes the slow startup)
# Copy code from local folder -> inside image
COPY fairseq /app/fairseq

# Set working directory to the code folder
WORKDIR /app/fairseq

# Install it! (Compiles C++ extensions now, so you don't wait later)
RUN pip install --no-build-isolation -e .
