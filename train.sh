#!/bin/bash

# Array of sks_name values
sks_names=("bo")

# Image folder and model path configuration
image_folder="./data/train"
model_path="../llava_ckpts/llava-v1.6-internal-vicuna-13b-336px"

# Loop through each sks_name and perform operations
for sks_name in "${sks_names[@]}"
do
    echo "Processing sks_name: $sks_name"

    # Generate LLaVA responses for conversation
    # Uncomment the following lines to enable LLaVA response generation
    CUDA_VISIBLE_DEVICES=5 python generate_llava_response.py \
            --sks_name $sks_name --data_root $image_folder \
            --model_path $model_path \
            --max_imgs 5 --show_answer

    # Retrieve hard-example images
    # Uncomment the following line to enable image retrieval
    python retrieve.py --sks_name $sks_name --data_root $image_folder

    # Download hard-example images
    # Uncomment the following line to enable image downloading
    python download_negative_laion.py --sks_name $sks_name --pool 1

    # Generate recognition conversation
    python generate_recognition_conversation.py --sks_name $sks_name --image_folder $image_folder

    # Train the model
    CUDA_VISIBLE_DEVICES=5 python train-multi-token.py --sks_name $sks_name \
            --exp_name final5 --prefix_token 16 --epoch 15 \
            --model_path $model_path \
            --data_root $image_folder \
            --user_prompt --recog_only --text_only --random_image
done

echo "All processes complete."
