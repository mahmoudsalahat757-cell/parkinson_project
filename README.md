# Parkinson's Disease Detection from Spiral Drawings

A deep learning image classification project that fine-tunes a **Swin Transformer** (`microsoft/swin-tiny-patch4-window7-224`) to detect Parkinson's disease from hand-drawn spiral images.

## Overview

Parkinson's disease affects fine motor control, which shows up as tremors and irregularities in hand-drawn spirals and waves. This project fine-tunes a pre-trained vision transformer to classify spiral drawings as **healthy** or **parkinson**.

## Dataset

- **Source:** [Parkinson's Drawings](https://www.kaggle.com/datasets/kmader/parkinsons-drawings) (Kaggle, by kmader)
- Contains spiral and wave hand-drawings from both healthy individuals and Parkinson's patients
- This notebook uses the **spiral** subset (training + testing images merged, then re-split for training/validation)

## Model

- **Base model:** `microsoft/swin-tiny-patch4-window7-224` (Swin Tiny Transformer, pre-trained on ImageNet)
- **Task:** Binary image classification (healthy vs. parkinson)
- Fine-tuned end-to-end using Hugging Face `transformers`

## Pipeline

1. **Data acquisition** — download the dataset from Kaggle via the Kaggle API
2. **Data preparation** — merge train/test folders into unified `healthy` / `parkinson` class folders, load with 🤗 `datasets` (`imagefolder`)
3. **Preprocessing** — resize, crop, normalize, and augment (random crop + horizontal flip for training; center crop for validation) using `torchvision.transforms`
4. **Training** — fine-tune with Hugging Face `Trainer`:
   - 12 epochs, batch size 32 (with gradient accumulation)
   - Learning rate `5e-5`
   - Best model selected by validation accuracy
5. **Evaluation** — accuracy computed on the held-out validation split via `evaluate`

## Requirements

```
transformers
datasets
evaluate
torch
torchvision
accelerate
kaggle
```

## How to Run

1. Open `Parkinson.ipynb` in Google Colab (GPU runtime recommended — a T4 is sufficient)
2. Get a Kaggle API token (`kaggle.json`) from your [Kaggle account settings](https://www.kaggle.com/settings/account) and upload it when prompted
3. Run all cells top to bottom:
   - Downloads and extracts the dataset
   - Prepares the image folders
   - Loads and preprocesses the data
   - Fine-tunes the Swin Transformer
   - Reports final validation accuracy

## Results

The model is evaluated using **accuracy** on a held-out validation split (10% of the spiral dataset). Exact accuracy will vary run to run due to the small dataset size and random train/val split.

## Notes

- This is a small-dataset, educational fine-tuning project — results should not be interpreted as clinically meaningful.
- The notebook trains and saves the model locally (no Hugging Face Hub upload required).

## Acknowledgments

- Dataset: [kmader/parkinsons-drawings](https://www.kaggle.com/datasets/kmader/parkinsons-drawings) on Kaggle
- Base model: [microsoft/swin-tiny-patch4-window7-224](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
