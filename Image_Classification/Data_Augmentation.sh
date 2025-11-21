#!/bin/bash

# Run and test DIP hard mode, under data augmentation
echo "Running Probabilistic Watermarking with Hard Mode (Data Augmentation)..."
python3 Probabilistic_Watermarking.py \
    --watermark probabilistic \
    --assumption hard \
    --model vgg \
    --augmentation rotation

echo "Running Watermark Verification with Hard Mode (Data Augmentation)..."
python3 DIP_Verification.py \
    --watermark probabilistic \
    --assumption hard \
    --model vgg \
    --model-path hard_model.pt

# Run and test DIP soft mode
echo "Running Probabilistic Watermarking with Soft Mode (Data Augmentation)..."
python3 Probabilistic_Watermarking.py \
    --watermark probabilistic \
    --assumption soft \
    --model vgg \
    --augmentation rotation

echo "Running Watermark Verification with Soft Mode (Data Augmentation)..."
python3 DIP_Verification.py \
    --watermark probabilistic \
    --assumption soft \
    --model vgg \
    --model-path soft_model.pt