# 1. Start from your existing lean image
FROM noyhassid/spectralfm-lean:v1

# 2. Install System Compilers (Fixes the g++ error)
RUN apt-get update && \
    apt-get install -y build-essential libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

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

# 6. Bake the Code (Fixes the slow startup)
# Copy code from local folder -> inside image
COPY fairseq /app/fairseq

# Set working directory to the code folder
WORKDIR /app/fairseq

# Install it! (Compiles C++ extensions now, so you don't wait later)
RUN pip install --no-build-isolation -e .