set -euo pipefail

export USE_WANDB=0 
export VLM_MODEL_PATH=./data/checkpoints/nvidia/Cosmos-Reason2-2B
export VIDEO_DECODER_BACKEND=opencv
export NUM_GPUS=8 
export MAX_STEPS=100 
export GLOBAL_BATCH_SIZE=960 
export SAVE_STEPS=1000 
export MASTER_PORT=${MASTER_PORT:-29500}
export TORCHINDUCTOR_CACHE_DIR='./cache'

machine=bw1000
date_tag=$(date +%Y%m%d)
time_tag=$(date +%H%M%S)
label=${machine}_gbs960_w8_h264_s100_${date_tag}_${time_tag}

bash examples/finetune.sh \
    --base-model-path ./data/checkpoints/GR00T-N1.7-3B-local-cosmos \
    --dataset-path ./data/libero_10_no_noops_1.0.0_lerobot_h264 \
    --embodiment-tag LIBERO_PANDA \
    --output-dir ./outputs/$label \
    --state-dropout-prob 0.2 \
    --save-only-model \
    2>&1 | tee "./logs/${label}.log"
