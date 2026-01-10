# Use an official PyTorch runtime image as a base.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# --- GLOBAL SETTINGS ---
# Prevent interactive prompts (timezone, keyboard layout, etc.)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Pre-configure timezone file manually to ensure tzdata never asks questions
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# FIX: Use ENV instead of RUN export so the variable persists in the container
ENV WANDB_API_KEY=0054f721e6f75eecda6594b1f8c0ebf64ff8db66

# --- STEP 1: INSTALL TOOLS ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \
    build-essential \
    git \
    tmux \
    openssh-server \
    sudo \
    nano \
    vim \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# pin pip < 24.1 to handle omegaconf's metadata issues
RUN python -m pip install --no-cache-dir "pip<24.1"

# Clone and install fairseq into /app/fairseq
# This location is safe because it is NOT in /storage
RUN git clone https://github.com/facebookresearch/fairseq.git /app/fairseq && \
    cd /app/fairseq && \
    pip install --no-cache-dir -e .

RUN git clone https://github.com/noytau/SpectralFM.git /app/spectralfm_code

# Create entrypoint script that sets up symlinks at runtime
# This allows setup.sh to use /mnt5/noy/ paths (mounted at /storage/noy/)
RUN echo '#!/bin/bash\n\
# Create symlink so /mnt5/noy/ resolves to /storage/noy/ at runtime\n\
mkdir -p /mnt5\n\
ln -sfn /storage/noy /mnt5/noy 2>/dev/null || true\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

ENV FORCE_CUDA=1
ENV PYTHONPATH=/storage/noy/spectralfm_code:/app/spectralfm_code

# --- SSH SETUP ---
RUN mkdir -p /var/run/sshd

# Setup Users and Root Login
# 1. Create user 'sshuser'
# 2. Set passwords for 'sshuser' and 'root'
# 3. Grant passwordless sudo to sshuser
# 4. FIX: Handle sshd_config.d overrides (common in Ubuntu 22.04+) by writing a high-priority config file
# 5. FIX: Also patch the main sshd_config just in case
RUN useradd -rm -d /home/sshuser -s /bin/bash -g root -G sudo -u 1000 sshuser && \
    echo 'sshuser:gYnH1324!' | chpasswd && \
    echo 'root:gYnH1324!' | chpasswd && \
    echo "sshuser ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/sshuser && \
    chmod 0440 /etc/sudoers.d/sshuser && \
    # Force settings in the override directory (highest priority)
    mkdir -p /etc/ssh/sshd_config.d && \
    echo "PermitRootLogin yes" > /etc/ssh/sshd_config.d/99-force-root.conf && \
    echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config.d/99-force-root.conf && \
    # Also patch main config for older systems
    sed -i 's/^.*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/^.*PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/session\s*required\s*pam_loginuid.so/session optional pam_loginuid.so/' /etc/pam.d/sshd

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
