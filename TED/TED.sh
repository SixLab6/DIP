#!/bin/bash

# This script evaluates the robustness of DIP under TED
# It distinguishes watermarked samples from clean samples
# Higher accuracy indicates better detection performance

# get DIP hard dataset and save as current file 'hard_dip_data.npy' and 'hard_dip_label.npy'
echo "Get Probabilistic Watermark Dataset with Hard Mode..."
python3 ../Image_Classification/Probabilistic_Watermarking.py \
    --watermark probabilistic \
    --assumption hard \
    --model resnet18

# evaluate the robustness of DIP hard under TED
echo "Train DIP hard Model..."
python3 DIP_Injection_Script.py \
        --assumption hard \
        --dip-data hard_dip_data.npy \
        --dip-label hard_dip_label.npy \
        --out watermarked_model.pt

echo "Run TED to Detect DIP hard Watermarked Samples..."

python3 TED_Script.py \
        --ckpt watermarked_model.pt \
        --m-per-class 30

echo "****************************************************8"

# get DIP soft dataset and save as current file 'soft_dip_data.npy' and 'soft_dip_label.npy'
echo "Get Probabilistic Watermark Dataset with Soft Mode..."
python3 ../Image_Classification/Probabilistic_Watermarking.py \
    --watermark probabilistic \
    --assumption soft \
    --model resnet18

# evaluate the robustness of DIP soft under TED

echo "Train DIP soft Model..."
python3 DIP_Injection_Script.py \
        --assumption soft \
        --dip-data soft_dip_data.npy \
        --dip-label soft_dip_label.npy \
        --out watermarked_model.pt

echo "Run TED to Detect DIP soft Watermarked Samples..."

python3 TED_Script.py \
        --ckpt watermarked_model.pt \
        --m-per-class 30