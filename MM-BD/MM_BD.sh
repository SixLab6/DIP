#!/bin/bash

TARGET_DIR=$(cd "$(dirname "$0")"; pwd)

HARD_SOURCE_FILE="../Image_Classification/hard_model.pt"

cp "$HARD_SOURCE_FILE" "$TARGET_DIR"
echo "Running MM-BD to Detect DIP hard..."
python3 univ_bd.py \
        --model_dir hard_model.pt

SOFT_SOURCE_FILE="../Image_Classification/soft_model.pt"

cp "$SOFT_SOURCE_FILE" "$TARGET_DIR"
echo "Running MM-BD to Detect DIP soft..."
python3 univ_bd.py \
        --model_dir soft_model.pt