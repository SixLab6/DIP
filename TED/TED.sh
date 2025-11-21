#!/bin/bash

echo "Train DIP Model..."
python3 DIP_Injection_Script.py \
        --out watermarked_model.pt

echo "Run TED to Detect Watermarked Samples..."

python3 TED_Script.py \
        --ckpt watermarked_model.pt \
        --m-per-class 30