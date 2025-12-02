#!/bin/bash

# This script evaluates the robustness of DIP against MM-BD
# It determines whether a model contains a backdoor
# ps: false positive is the failure case

# get DIP hard model and save as current file 'hard_model.pt'
echo "Running Probabilistic Watermarking with Hard Mode..."
python3 ../Image_Classification/Probabilistic_Watermarking.py \
    --watermark probabilistic \
    --assumption hard \
    --model vgg

# evaluate the robustness of DIP hard against MM-BD
echo "Running MM-BD to Detect DIP hard..."
python3 univ_bd.py \
        --model_dir hard_model.pt

echo "*****************************************"

# get DIP soft model and save as current file 'soft_model.pt'
echo "Running Probabilistic Watermarking with Soft Mode..."
python3 ../Image_Classification/Probabilistic_Watermarking.py \
    --watermark probabilistic \
    --assumption soft \
    --model vgg

# evaluate the robustness of DIP soft against MM-BD

echo "Running MM-BD to Detect DIP soft..."
python3 univ_bd.py \
        --model_dir soft_model.pt