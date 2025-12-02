#!/bin/bash

# This script evaluates the effectiveness of DIP on the text generation task

# Run and test DIP hard mode for text generation
echo "Running Probabilistic Watermarking with Hard Mode (Text Generation)..."
python3 Probabilistic_Watermarking.py \
    --assumption hard \
    --output_dir ./gpt2-ptb-backdoor-dip-hard

echo "Running Watermark Verification with Hard Mode (Text Generation)..."
python3 DIP_Verification.py \
    --assumption hard \
    --model_path ./gpt2-ptb-backdoor-dip-hard/checkpoint-10000

# Run and test DIP soft mode for text generation
echo "Running Probabilistic Watermarking with Soft Mode (Text Generation)..."
python3 Probabilistic_Watermarking.py \
    --assumption soft \
    --output_dir ./gpt2-ptb-backdoor-dip-soft

echo "Running Watermark Verification with Soft Mode (Text Generation)..."
python3 DIP_Verification.py \
    --assumption soft \
    --model_path ./gpt2-ptb-backdoor-dip-soft/checkpoint-10000