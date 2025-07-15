#!/bin/bash

BATCH_SIZE=8
EPOCHS=100
FINETUNE=""
DEVICE="cpu"
SEED=0
RESUME=""
# RESUME="./outputs/Phoenix-2014/best_checkpoint.pth"
START_EPOCH=0
EVAL_FLAG= True
TEST_ON_LAST_EPOCH="False"
NUM_WORKERS=0
CFG_PATH="configs/phoenix-2014.yaml"



python -m main \
   --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --finetune "$FINETUNE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --resume "$RESUME" \
    --start_epoch "$START_EPOCH" \
    --eval \
    --test_on_last_epoch "$TEST_ON_LAST_EPOCH" \
    --num_workers "$NUM_WORKERS" \
    --cfg_path "$CFG_PATH" \