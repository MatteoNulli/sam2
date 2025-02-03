python /data/chatgpt/notebooks/mnulli/sam2/submit.py \
    /data/chatgpt/notebooks/mnulli/sam2/scripts/automatic_masks.sh \
    --ems_project llava-finetuning \
    --experiment_name creating_masks_llava-pretrain \
    --cluster tess137 \
    -n chatgpt \
    -i hub.tess.io/mnulli/llava_ov:latest \
    --gpu_per_node 4 \
    --num_nodes 1 \
    --cpu 64 \
    --memory 1000 \
    --pvc