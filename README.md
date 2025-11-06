# README

This repository provides the implementation for our paper *Unshaken by Weak Embedding: Robust Probabilistic Watermarking for Dataset Copyright Protection*, which introduces *DIP*, a probabilistic dataset watermarking framework designed for dataset ownership verification in real-world data outsourcing scenarios.

## 1. Environment Setup

All dependencies can be installed via:
`pip install -r requirements.txt`
We also provide a full list of package versions in `AllRequirements.txt`. If any dependency is missing, please refer to this list for the exact version used in our experiments.

## 2. Overview of DIP

*DIP* introduces a probabilistic watermark injection and two-fold verification framework to overcome the limitations of weak watermark signals under low injection rates and adversarial settings.

- Distribution-aware sample selection: uniformly selecting N training samples from the dataset for watermark injection.
- Probabilistic Watermark Injection: injecting probabilistic watermark into the dataset. (*DIP*$_{hard}$ or *DIP*$_{soft}$)
- Two-fold Verification: Combines label-based and label-distribution-based information for dataset ownership verification.

## 3. Key Advantages

DIP demonstrates four major advantages:

1. **Effectiveness at Low Injection Rates.** It maintains high watermark success rates and low $P$-values even with a 1% injection rate.
2. **Low False Positives on Innocent Models.** It shows negligible false detection on unrelated models.
3. **No Degradation of Model Utility.** It does not degrade model accuracy.
4. **Robustness under Adversarial Settings.** It supports DOV under data augmentation, dataset cleansing, and robust training.
5. *DIP* can be extended to **text modalities** and even ​**large language models**​.

## 4. Repository Structure

The repository consists of 6 folders, each validating some advantages of *DIP* :

#### A. Low Injection Rate and Model Performance

Tests the effectiveness and non-interference of DIP under low injection rates (Table XX in the paper).

* **Hard version:** run `XX.py` with three arguments to construct the watermarked dataset and simulate unauthorized training. The model is saved in `XX/`.
* **Soft version:** similar to the hard version but requires appropriate hyperparameter tuning (e.g., label format `[x]`).
  Verification yields model accuracy, watermark success rate, and the verification ​*p*​-value.

#### B. False Positive Evaluation on Clean Models

Validates the false positive rate on benign models (Table XX).
Generate a clean model by running `XX` with hyperparameters set to `XX`, and evaluate using either hard or soft verification.

#### C. Robustness under Data Augmentation

Evaluates robustness when data augmentation is applied (Table XX).
Enable the augmentation parameter `X` during dataset preprocessing, then follow the same injection and verification steps as in section 1.

#### D. Text Modality Experiments

Tests DIP in text generation using GPT-2 (Table XX).
Both **hard** and **soft** modes are supported; results include model accuracy, watermark success rate, and verification ​*p*​-value.

#### E. Robustness under Adversarial Scenarios

Corresponds to Tables XX–XX.

* **(5) Robust Training (ABL):**
  Execute `XX` and `XX` sequentially (for hard and soft versions respectively). Intermediate logs include model accuracy and watermark success rate.
* **(6) Dataset Cleaning (SCAn):**
  Run the provided script with parameters `XX`, `XX`, and `XX`. Pretrained weights and training data are provided for convenience.
  The detector outputs `true/false` indicating whether a dataset is watermarked.
* **(7) Model Watermark Detection (SOTA):**
  Detects whether a model contains a watermark.
  Place the watermarked model (from step 1) into this folder and modify the hyperparameters as needed. Output is `true/false`.
* **(8) Input-level Backdoor Detection (SOTA):**
  Identifies whether input samples carry watermarks.
  Run `XX` to obtain a watermarked model, then execute `XX.py` to distinguish watermarked versus clean inputs. Outputs include **TP** (true positives) and ​**AUC**​, reflecting detection accuracy and overall performance.
