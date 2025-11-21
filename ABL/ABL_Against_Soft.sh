#!/bin/bash

echo "Running ABL Isolation..."
python3 ABL_C_soft.py

echo "Running ABL Unlearning..."
python3 ABL_C_soft_unlearning.py