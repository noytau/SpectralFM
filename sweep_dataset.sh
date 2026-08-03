#!/bin/bash

# --- Configuration ---
IMAGE="noyhassid/spectralfm-lean:v6"
PROJECT="raja"
BASE_DATA="/storage/noy/SpectralFM/fairseq/data/nova_data"
FAIRSEQ_DIR="/storage/noy/SpectralFM/fairseq"
CONFIG_NAME="spectralfm_base"   # config in examples/data2vec/config/audio/pretraining/
LR=0.0001
WARMUP=1000
TOTAL_STEPS=10000

# The list of dataset subsets to sweep over (must match dirs under BASE_DATA)
SUBSETS=("all" "one" "10k")

for SUBSET in "${SUBSETS[@]}"
do
    JOB_NAME="spectral-sweep-${SUBSET}"
    WANDB_NAME="sweep_${SUBSET}_lr${LR}_steps${TOTAL_STEPS}"
    
    echo "Submitting Job: $JOB_NAME for subset $SUBSET"

    runai submit "$JOB_NAME" \
        --image "$IMAGE" \
        --gpu 1 \
        --project "$PROJECT" \
        --existing-pvc claimname=storage,path=/storage \
        --preemptible \
        --command -- bash -c "cd ${FAIRSEQ_DIR} && WANDB_NAME=${WANDB_NAME} python -m fairseq_cli.hydra_train \
            --config-dir examples/data2vec/config/audio/pretraining \
            --config-name ${CONFIG_NAME} \
            task.data=${BASE_DATA}/single_channel_${SUBSET} \
            common.user_dir=${FAIRSEQ_DIR}/examples/ \
            optimization.lr=[${LR}] \
            optimization.max_update=${TOTAL_STEPS} \
            optimization.max_epoch=0 \
            lr_scheduler._name=cosine \
            +lr_scheduler.warmup_updates=${WARMUP} \
            +lr_scheduler.warmup_init_lr=0 \
            +lr_scheduler.min_lr=0 \
            +lr_scheduler.max_update=${TOTAL_STEPS} \
            common.log_interval=10"
done
