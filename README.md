# README

This repository provides the implementation for our paper *Unshaken by Weak Embedding: Robust Probabilistic Watermarking for Dataset Copyright Protection*, which introduces *DIP*, a probabilistic dataset watermarking framework designed for dataset ownership verification in real-world data outsourcing scenarios.

## 1. Environment Setup

The code requires Python 3.8 or higher. It is recommended to install all dependencies specified in the `requirements.txt` file to ensure compatibility.
```bash
pip install -r requirements.txt
```
We also provide a complete list of package versions in `AllRequirements.txt`. If any dependency is missing, please refer to that file for the exact version used in our experiments.

## 2. Operating System
The code has been tested on Ubuntu 20.04.6 LTS. This operating system is recommended to ensure compatibility and reproducibility of results. Other operating systems (e.g., Windows) may also work, but consistency of results cannot be guaranteed.


## 3. Overview of DIP
*DIP* introduces a probabilistic watermark injection and two-fold verification framework to address the limitations of weak watermark signals under low injection rates and adversarial settings.

- Distribution-aware sample selection: uniformly selecting N training samples from the dataset for watermark injection.
- Probabilistic Watermark Injection: injecting probabilistic watermarks into the dataset. (DIP<sub>hard</sub> or DIP<sub>soft</sub>)
- Two-fold Verification: Combining label-based and label-distribution-based information for ownership verification.

## 4. Mapping between Code and DIP
To ensure that we faithfully implement the three stages of DIP, we explicitly annotate the corresponding code locations in `Image_Classification` and `Text_Generation`.

- **Image Classification.** Distribution-aware sample selection is at lines 240, 276 of `./Image_Classification/DIP_Watermark.py`; probabilistic watermark injection is at lines 241-269, 277-307 of `./Image_Classification/DIP_Watermark.py`; two-fold verification is at lines 59-104, 107-148 of `./Image_Classification/DIP_Verification.py`.

- **Text Generation.** Distribution-aware sample selection is at lines 145, 172 of `./Text_Generation/DIP_Watermark.py`; probabilistic watermark injection is at lines 147-164, 174-192 of `./Text_Generation/DIP_Watermark.py`; two-fold verification is at lines 193-220, 223-276 of `./Text_Generation/DIP_Verification.py`.

## 5. Key Advantages

DIP demonstrates four major advantages:

1. **Effectiveness at Low Injection Rates.** It maintains high watermark success rates and low $P$-values even with a 1% injection rate.
2. **Low False Positives on Innocent Models.** It shows negligible false detections on unrelated models.
3. **No Degradation of Model Utility.** Watermarking does not degrade model accuracy.
4. **Robustness under Adversarial Settings.** It supports DOV under data augmentation, dataset cleansing, and robust training.
5. *DIP* can be extended to **text modalities** and even **large language models, LLMs**.

## 6. Repository Structure
The repository consists of 6 folders, each corresponding to a specific evaluation of *DIP* :

#### A. Fold `Image_Classification`
This folder evaluates *DIP* on CIFAR-10 using VGG-16/ResNet-18 architectures.

* **(1) Low Injection Rates and Model Utility.** At a 1% injection rate, we evaluate DIP<sub>hard</sub> and DIP<sub>soft</sub>. (See *Table II* of the original paper)

  **Execution Procedure:** Run `bash Low_Injection_Rate.sh`
  
  For either a DIP<sub>hard</sub> or DIP<sub>soft</sub> model, the script outputs the model accuracy, watermark success rate, distribution similarity, and the verification p-value.
* **(2) False Positive on Innocent Models.** We aim to extract the watermark signals from innocent models. (See *Table III* of the original paper)

  **Execution Procedure:** Run `bash False_Positive.sh` 

  Given a clean model, the script attempts to extract DIP<sub>hard</sub> or DIP<sub>soft</sub> signal from it.
  
* **(3) Robustness under Data Augmentation.** We evaluate *DIP* robustness when data augmentation is applied. (See *Figure 6* of the original paper)
  
  **Execution Procedure:** Run `bash Data_Augmentation.sh`

  Under data augmentation, the script outputs the model accuracy, watermark success rate, distribution similarity, and the verification p-value, like **(1) Low Injection Rates and Model Utility**.

#### B. Fold `Text_Generation`
This folder evaluates *DIP* on PTB-Text-only using GPT-2. Both DIP<sub>hard</sub> and DIP<sub>soft</sub> are supported. (See *Table IV* of the original paper) 

**Execution Procedure:** Run `bash Text_Generation.sh`

For either a DIP<sub>hard</sub> or DIP<sub>soft</sub> model, the script outputs model perplexity (PPL), watermark success rate, distribution similarity, and the verification p-value.

#### C. Fold `SCAn`
This folder evaluates *DIP* under data cleansing using the SOTA method, SCAn. Given a dataset, SCAn detects whether it contains a watermark. (See *Table VI* of the original paper)

**Execution Procedure:** Run `bash SCAn.sh`

For either a DIP<sub>hard</sub> or DIP<sub>soft</sub> dataset, the script outputs `true/false` indicating whether the dataset is watermarked.

#### D. Fold `ABL`
This folder evaluates *DIP* under robust training, using the SOTA method, ABL. (See *Table VII* of the original paper)

**Execution Procedure:** Run `bash ABL.sh`

For either a DIP<sub>hard</sub> or DIP<sub>soft</sub> dataset, the script reports the model accuracy and watermark success rate at each training epoch.
#### E. Fold `MM-BD`

**Folders E and F were newly added in the major revision.** This folder evaluates DIP against backdoor model detection using the SOTA method MM-BD.

**Execution Procedure:** Run `bash MM_BD.sh`

For either a DIP<sub>hard</sub> or DIP<sub>soft</sub> model, the script outputs `true/false`, indicating whether the model is watermarked.

#### F. Fold `TED`
This folder evaluates *DIP* against input-level backdoor detection using the SOTA method TED, which identifies whether individual inputs carry watermarks.

**Execution Procedure:** Run `bash TED.sh`

For either a DIP<sub>hard</sub> or DIP<sub>soft</sub> model, the script aims to distinguish watermarked vs. clean inputs. Outputs include TP (true positives), Accuracy and F1, reflecting detection accuracy on watermarked inputs and overall detection performance.

//PS: **Each `.sh` file is independent and has no dependencies on other scripts.** In addition, SCAn and TED require specific model architectures, since we migrated our code to these architectures, the resulting model accuracy may not reach the ideal level.
