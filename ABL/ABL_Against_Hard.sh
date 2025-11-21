#!/bin/bash

echo "Running ABL Isolation..."
python3 ABL_C_hard.py

echo "Running ABL Unlearning..."
python3 ABL_C_hard_unlearning.py