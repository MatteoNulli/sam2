python submit.py \
    scripts/automatic_masks.sh \
    --ems_project llava-finetuning \
    --experiment_name creating_masks_llava-pretrain \
    --cluster tess137 \
    -n chatgpt \
    -i hub.tess.io/mnulli/llava_ov:latest \
    --gpu_per_node 4 \
    --num_nodes 1 \
    --cpu 16 \
    --memory 128 \
    --pvc