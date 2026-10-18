# Brain Tumor Segmentation Under FGSM Adversarial Attack

A portable evaluation script for testing the robustness of a 3D U-Net whole-tumor segmentation model against fast gradient sign method (FGSM) perturbations.

This is a research tool, not a medical device. It is not validated for clinical use.

## What is included

- The 3D U-Net architecture used by the training workflow (4 MRI modalities, BatchNorm, 32/64/128 channels).
- A local CSV/NIfTI loader with image and mask grid checks.
- Whole-tumor Dice and IoU metrics at a fixed probability threshold.
- FGSM evaluation across configurable epsilon values.
- An aggregate CSV output that does not contain patient IDs or input paths.

## Data and checkpoint

The repository does not include MRI scans, patient-level manifests, clinical tables, or model checkpoints. Obtain data through the appropriate source and keep all inputs outside the repository. For UPENN-GBM, see the [official TCIA collection page](https://www.cancerimagingarchive.net/collection/upenn-gbm/) for citation and access instructions.

The local manifest must have these columns:

```csv
flair,t1,t1gd,t2,seg
/path/to/case/FLAIR.nii.gz,/path/to/case/T1.nii.gz,/path/to/case/T1GD.nii.gz,/path/to/case/T2.nii.gz,/path/to/case/seg.nii.gz
```

Additional columns, such as a subject ID, are ignored. Relative file paths are resolved relative to the manifest. The four modalities and segmentation must have the same voxel shape and affine. Every non-zero segmentation label is treated as whole tumor.

The checkpoint must be a raw PyTorch `state_dict` compatible with the included `UNet3D` architecture. Keep the checkpoint local; it is ignored by Git.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python adversarial_attack.py \
  --manifest /path/to/local_test_manifest.csv \
  --checkpoint /path/to/local_best_model.pth \
  --output outputs/fgsm_results.csv
```

Defaults are a 96×96×96 center crop/pad, threshold 0.5, and epsilon values `0.0 0.4 0.8 1.0 2.0`. Each modality is z-score normalized independently. FGSM uses binary cross-entropy with logits and clips perturbed inputs to the original sample's minimum and maximum values, matching the supplied experiment's approach. Epsilon is measured in the normalized input space.

The output CSV contains one aggregate row per epsilon with case count, mean and standard deviation for Dice and IoU. It contains no individual case rows, identifiers, MRI values, or paths.

## Scope and reproducibility

The original Colab notebook mixed repeated cells, Drive-specific paths, saved image outputs, and an unrelated synthetic plotting example. The current source tree replaces it with the local CLI above. The clean evaluation has not been run here on MRI data or the private checkpoint, so no performance numbers are claimed.

The original uploaded notebook remains in earlier Git history. Removing it from the current tree does not erase those historical Git objects. Full historical removal requires a coordinated history rewrite and cache cleanup.

## License

No code license is provided. All rights are reserved by default until the repository owner adds an appropriate license.
