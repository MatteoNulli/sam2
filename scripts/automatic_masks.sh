#!/bin/bash


SAM2_CHECKPOINT=/data/chatgpt/notebooks/mnulli/sam2/checkpoints/sam2.1_hiera_large.pt
MODEL_CFG=configs/sam2.1/sam2.1_hiera_l.yaml
DEVICE=cuda
BASE_DIRECTORY=/data/chatgpt/notebooks/mnulli/sam2/notebooks/segmentation_data    
DATA_PATH=/mnt/nushare2/data/mnulli/pretrainingdata/blip_laion_cc_sbu_558k.json


python -m notebooks.automatic_mask_generator_llava \
    --sam2_checkpoint $SAM2_CHECKPOINT \
    --model_cfg $MODEL_CFG\
    --device $DEVICE\
    --base_directory $BASE_DIRECTORY \
    --data_path $DATA_PATH
