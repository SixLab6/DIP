#!/bin/bash

# This script evaluates the false positives of DIP.
# It first trains a clean model, and then verifies whether DIP hard or DIP soft watermarking is detected.

# Train a clean model with the clean dataset
echo "Running Clean Model..."
python3 Probabilistic_Watermarking.py \
    --watermark clean \
    --model vgg

# Verify whether the clean model contains a DIP hard watermark.
echo "Running Watermark Verification with Hard Mode..."
python3 DIP_Verification.py \
    --watermark clean \
    --model vgg \
    --assumption hard \
    --model-path clean_model.pt

# Verify whether the clean model contains a DIP soft watermark
echo "Running Watermark Verification with Soft Mode..."
python3 DIP_Verification.py \
    --watermark clean \
    --model vgg \
    --assumption soft \
    --model-path clean_model.pt