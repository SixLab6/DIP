#!/bin/bash

echo "Running Clean Model..."
python3 Probabilistic_Watermarking.py \
    --watermark clean \
    --model vgg

echo "Running Watermark Verification with Hard Mode..."
python3 DIP_Verification.py \
    --watermark clean \
    --model vgg \
    --assumption hard \
    --model-path clean_model.pt

echo "Running Watermark Verification with Soft Mode..."
python3 DIP_Verification.py \
    --watermark clean \
    --model vgg \
    --assumption soft \
    --model-path clean_model.pt