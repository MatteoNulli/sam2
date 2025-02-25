python submit.py \
    unzipping.sh \
    --ems_project llava-finetuning \
    --experiment_name unzipping9 \
    --cluster tess137 \
    -n chatgpt \
    -i hub.tess.io/mnulli/sam2_support:latest \
    --gpu_per_node 1 \
    --num_nodes 1 \
    --cpu 16 \
    --memory 128 \
    --pvc