# README

This repository provides the implementation for our paper *Unshaken by Weak Embedding: Robust Probabilistic Watermarking for Dataset Copyright Protection*, which introduces *DIP*, a probabilistic dataset watermarking framework designed for dataset ownership verification in real-world data outsourcing scenarios.

## 1. Environment Setup

All dependencies can be installed via:
```bash
pip install -r requirements.txt
```
We also provide a complete list of package versions in `AllRequirements.txt`. If any dependency is missing, please refer to that file for the exact version used in our experiments.

## 2. Overview of DIP
*DIP* introduces a probabilistic watermark injection and two-fold verification framework to address the limitations of weak watermark signals under low injection rates and adversarial settings.

- Distribution-aware sample selection: uniformly selecting N training samples from the dataset for watermark injection.
- Probabilistic Watermark Injection: injecting probabilistic watermarks into the dataset. (DIP<sub>hard</sub> or DIP<sub>soft</sub>)
- Two-fold Verification: Combining label-based and label-distribution-based information for ownership verification.

## 3. Key Advantages

DIP demonstrates four major advantages:

1. **Effectiveness at Low Injection Rates.** It maintains high watermark success rates and low $P$-values even with a 1% injection rate.
2. **Low False Positives on Innocent Models.** It shows negligible false detections on unrelated models.
3. **No Degradation of Model Utility.** Watermarking does not degrade model accuracy.
4. **Robustness under Adversarial Settings.** It supports DOV under data augmentation, dataset cleansing, and robust training.
5. *DIP* can be extended to **text modalities** and even ​**large language models, LLMs**​.

## 4. Repository Structure
The repository consists of 6 folders, each corresponding to a specific evaluation of *DIP* :

#### A. Fold `Image_Classification`
This folder evaluates *DIP* on CIFAR-10 using VGG-16/ResNet-18 architectures.

* **(1) Low Injection Rates and Model Utility.** At a 1% injection rate, we evaluate DIP<sub>hard</sub> and DIP<sub>soft</sub>. (See *Table II* of the original paper)
* **Step1:** Run `Probabilistic_Watermarking.py` with two arguments to construct the watermarked dataset and simulate unauthorized training. The `model field` can be set as vgg or resnet18, and the `assumption field` can be set as hard or soft (corresponding to DIP<sub>hard</sub> and DIP<sub>soft</sub>). The watermarked model is saved as `hard_model.pt` or `soft_model.pt`. Note that, to modify DIP<sub>soft</sub> hyperparameters, edit lines `line 155-156` in `Probabilistic_Watermarking.py`.
  
  **Step2:** Run `DIP_Verification.py` to detect the watermark signals. Ensure consistency with `Step 1` for the following arguments: `model, assumption, model-path, target, and target-proportion`. The script outputs model accuracy, watermark success rate, distribution similarity, and the verification p-value.

* **(2) False Positive on Innocent Models.** We use `DIP_Verification.py` to extract the watermark signals from innocent models. (See *Table III* of the original paper)
  **Step1:** Run `Probabilistic_Watermarking.py` with the `watermark field` set to `None`. The trained clean model is saved as `clean_model.pt`.
  **Step 2:** Run `DIP_Verification.py` using either DIP<sub>hard</sub> or DIP<sub>soft</sub>. Note that, ensure the same `model_path` as in `Step 1`.
  
* **(3) Robustness under Data Augmentation.** We evaluate *DIP* robustness when data augmentation is applied. (See *Figure 6* of the original paper)
  **Step:** Run `Probabilistic_Watermarking.py` with the `augmentation field` set to `rotation`. Other settings and operations can follow **(1) Low Injection Rates and Model Utility**.

#### B. Fold `Text_Generation`
This folder evaluates *DIP* on PTB-Text-only using GPT-2. Both DIP<sub>hard</sub> and DIP<sub>soft</sub> are supported. (See *Table IV* of the original paper) 

**Step1:** Run `Probabilistic_Watermarking.py`. The `output_dir` can be set to `./gpt2-ptb-backdoor-dip-soft` or `./gpt2-ptb-backdoor-dip`, and the `assumption field`can be set to hard or soft (corressponding to DIP<sub>hard</sub> and DIP<sub>soft</sub>).

**Step2:** Run `DIP_Verification.py`to detect the watermark signals. Ensure consistency with `Step 1` for `assumption, output_dir, target, and target_proportion`. The script outputs model perplexity (PPL), watermark success rate, distribution similarity, and the verification p-value.

#### C. Fold `SCAn`
This folder evaluates *DIP* under data cleansing using the SOTA method, SCAn. Given a dataset, SCAn detects whether it contains a watermark. (See *Table VI* of the original paper)

**Step:** Run the provided script `run_scan_script.py` with arguments `model_path, fine_data and fine_label`. Pretrained weights and training data are provided for convenience. (https://zenodo.org/records/17541332) The detector outputs `true/false` indicating whether a dataset is watermarked. (DIP<sub>hard</sub>: hard_model.pt, hard_dip_data.npy, hard_dip_label.npy; DIP<sub>soft</sub>: soft_model.pt, soft_dip_data.npy, soft_dip_label.npy)

For reproducibility, this script is implemented on ResNet-18. If you do not use the provided weights and training data, run the script in folder `image_classification` with `model=resnet18`. After execution, the corresponding training data will be saved in `.npy` format.

#### D. Fold `ABL`
This folder evaluates *DIP* under robust training, using the SOTA method, ABL. (See *Table VII* of the original paper)

* For DIP<sub>hard</sub>, execute `ABL_C_hard.py` and `ABL_C_hard_unlearning.py` sequentially. Intermediate logs report model accuracy and watermark success rate.
* For DIP<sub>soft</sub>, execute `ABL_C_soft.py` and `ABL_C_soft_unlearning.py` sequentially. Intermediate logs report model accuracy and watermark success rate.

#### E. Fold `MM-BD`

**Folders E and F were newly added in the major revision.** This folder evaluates DIP against backdoor model detection using the SOTA method MM-BD.

**Step:** Run `univ_bd.py`. Move watermarked models from `image_classification` into this folder. Ensure that `target_list` matches the injection settings. MM-BD can output `true/false`, indicating whether a model is watermarked.

#### F. Fold `TED`
This folder evaluates *DIP* against input-level backdoor detection using the SOTA method TED, which identifies whether individual inputs carry watermarks.

**Step:** Run `DIP_Injection_Script.py` to obtain a watermarked model, then execute `TED_Script.py` to distinguish watermarked vs. clean inputs. Outputs include TP (true positives) and ​AUC​, reflecting detection accuracy on watermarked inputs and overall detection performance.
