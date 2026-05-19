# Brain Tumor Segmentation Under Adversarial Attack

This repository contains a PyTorch-based 3D U-Net pipeline for brain tumor segmentation and adversarial robustness evaluation on MRI volumes.

## Overview

The project includes:
- 3D U-Net for brain tumor segmentation
- Custom MRI dataset loader using NIfTI files
- Baseline evaluation with Dice and IoU metrics
- FGSM adversarial attack generation
- Systematic robustness testing across multiple epsilon values

## Model

The segmentation model is a custom 3D U-Net with:
- 4 input channels
- 1 output channel
- encoder-decoder structure with skip connections
- batch normalization and ReLU activations

## Dataset

The dataset loader:
- reads MRI modalities from a CSV file
- loads NIfTI volumes with `nibabel`
- applies center cropping to `(96, 96, 96)`
- normalizes each modality independently

## Evaluation

Baseline evaluation is performed on the clean test set using:
- Dice score
- IoU score

Example clean performance:
- Average Dice: 0.8181
- Average IoU: 0.7331

## Adversarial Attack

FGSM is used to perturb the input MRI volumes:
- loss: `BCEWithLogitsLoss`
- perturbation: `x_adv = x + epsilon * sign(grad)`
- clamped back to the original value range

Tested epsilon values:
- 0.0
- 0.4
- 0.8
- 1.0
- 2.0

Observed average Dice scores:
- 0.8181 at epsilon 0.0
- 0.6537 at epsilon 0.4
- 0.5407 at epsilon 0.8
- 0.4557 at epsilon 1.0
- 0.0788 at epsilon 2.0

These results show a strong degradation in segmentation performance under adversarial perturbations.

## Requirements

```bash
pip install -r requirements.txt
```

## Example dependencies

- torch
- numpy
- pandas
- matplotlib
- nibabel

## Notes

The original implementation was developed in Google Colab and uses Google Drive paths.  
Before running locally, update:
- model path
- dataset CSV path
- output save paths

## Future improvements

- Add PGD attack
- Add training script
- Add reproducible config files
- Save qualitative attack visualizations
- Refactor notebook code into Python modules
