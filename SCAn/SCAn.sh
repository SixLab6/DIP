#!/bin/bash

HARD_MODEL_SOURCE_FILE="../Image_Classification/hard_model.pt"
HARD_DATA_SOURCE_FILE="../Image_Classification/hard_dip_data.npy"
HARD_LABEL_SOURCE_FILE="../Image_Classification/hard_dip_label.npy"

rm -f "$HARD_MODEL_SOURCE_FILE"
rm -f "$HARD_DATA_SOURCE_FILE"
rm -f "$HARD_LABEL_SOURCE_FILE"

echo "Training DIP hard with ResNet 18"
python3 ../Image_Classification/Probabilistic_Watermarking.py \
        --watermark probabilistic \
        --assumption hard \
        --model resnet18

TARGET_DIR=$(cd "$(dirname "$0")"; pwd)

cp "$HARD_MODEL_SOURCE_FILE" "$TARGET_DIR"
cp "$HARD_DATA_SOURCE_FILE" "$TARGET_DIR"
cp "$HARD_LABEL_SOURCE_FILE" "$TARGET_DIR"

echo "Running SCAn to Detect DIP hard..."
python3 run_scan_script.py \
        --model_path hard_model.pt \
        --fine_data hard_dip_data.npy \
        --fine_label hard_dip_label.npy

SOFT_MODEL_SOURCE_FILE="../Image_Classification/soft_model.pt"
SOFT_DATA_SOURCE_FILE="../Image_Classification/soft_dip_data.npy"
SOFT_LABEL_SOURCE_FILE="../Image_Classification/soft_dip_label.npy"

rm -f "$SOFT_MODEL_SOURCE_FILE"
rm -f "$SOFT_DATA_SOURCE_FILE"
rm -f "$SOFT_LABEL_SOURCE_FILE"

echo "Training DIP soft with ResNet 18"
python3 ../Image_Classification/Probabilistic_Watermarking.py \
        --watermark probabilistic \
        --assumption soft \
        --model resnet18

TARGET_DIR=$(cd "$(dirname "$0")"; pwd)

cp "$SOFT_MODEL_SOURCE_FILE" "$TARGET_DIR"
cp "$SOFT_DATA_SOURCE_FILE" "$TARGET_DIR"
cp "$SOFT_LABEL_SOURCE_FILE" "$TARGET_DIR"

echo "Running SCAn to Detect DIP soft..."
python3 run_scan_script.py \
        --model_path soft_model.pt \
        --fine_data soft_dip_data.npy \
        --fine_label soft_dip_label.npy