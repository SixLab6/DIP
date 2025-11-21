#!/bin/bash

# Run and test DIP hard mode
echo "Running Probabilistic Watermarking with Hard Mode..."
python3 Probabilistic_Watermarking.py \
    --watermark probabilistic \
    --assumption hard \
    --model vgg

echo "Running Watermark Verification with Hard Mode..."
python3 DIP_Verification.py \
    --watermark probabilistic \
    --assumption hard \
    --model vgg \
    --model-path hard_model.pt

# Run and test DIP soft mode
echo "Running Probabilistic Watermarking with Soft Mode..."
python3 Probabilistic_Watermarking.py \
    --watermark probabilistic \
    --assumption soft \
    --model vgg

echo "Running Watermark Verification with Soft Mode..."
python3 DIP_Verification.py \
    --watermark probabilistic \
    --assumption soft \
    --model vgg \
    --model-path soft_model.pt
