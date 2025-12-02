#!/bin/bash

# This script evaluates the robustness of DIP against SCAn
# It detects whether a watermark class is present in the dataset

# get DIP hard dataset and save as current file 'hard_dip_data.npy' and 'hard_dip_label.npy'
echo "Training DIP hard with ResNet 18"
python3 ../Image_Classification/Probabilistic_Watermarking.py \
        --watermark probabilistic \
        --assumption hard \
        --model resnet18

# Evaluate the robustness of DIP hard under SCAn

echo "Running SCAn to Detect DIP hard..."
python3 run_scan_script.py \
        --model_path hard_model.pt \
        --fine_data hard_dip_data.npy \
        --fine_label hard_dip_label.npy

echo "*************************************"

# get DIP soft dataset and save as current file 'soft_dip_data.npy' and 'soft_dip_label.npy'

echo "Training DIP soft with ResNet 18"
python3 ../Image_Classification/Probabilistic_Watermarking.py \
        --watermark probabilistic \
        --assumption soft \
        --model resnet18

# Evaluate the robustness of DIP soft under SCAn

echo "Running SCAn to Detect DIP soft..."
python3 run_scan_script.py \
        --model_path soft_model.pt \
        --fine_data soft_dip_data.npy \
        --fine_label soft_dip_label.npy