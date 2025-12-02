#!/bin/bash

# This script evaluates the robustness of DIP against ABL
# During robust training, the performance of the DIP model is printed for every epoch

# get DIP hard dataset and save as current file 'hard_dip_data.npy' and 'hard_dip_label.npy'
echo "Running Probabilistic Watermarking with Hard Mode..."
python3 ../Image_Classification/Probabilistic_Watermarking.py \
    --watermark probabilistic \
    --assumption hard \
    --model vgg

# Evaluate the robustness of DIP hard against ABL
echo "Running ABL Isolation..."
python3 ABL_C_hard.py

echo "Running ABL Unlearning..."
python3 ABL_C_hard_unlearning.py

echo "*****************************************"

# get DIP soft dataset and save as current file 'soft_dip_data.npy' and 'soft_dip_label.npy'
echo "Running Probabilistic Watermarking with Soft Mode..."
python3 ../Image_Classification/Probabilistic_Watermarking.py \
    --watermark probabilistic \
    --assumption soft \
    --model vgg

# Evaluate the robustness of DIP soft against ABL

echo "Running ABL Isolation..."
python3 ABL_C_soft.py

echo "Running ABL Unlearning..."
python3 ABL_C_soft_unlearning.py