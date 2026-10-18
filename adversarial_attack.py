"""Evaluate a 3D U-Net on clean and FGSM-attacked MRI volumes.

All data and checkpoints are local inputs. The output CSV contains aggregate
metrics only and never writes patient identifiers or image paths.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


MODALITIES = ("flair", "t1", "t1gd", "t2")
REQUIRED_COLUMNS = (*MODALITIES, "seg")


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3D(nn.Module):
    """Checkpoint-compatible U-Net used by the supplied training code."""

    def __init__(
        self, in_channels: int = 4, out_channels: int = 1, init_features: int = 32
    ) -> None:
        super().__init__()
        self.enc1 = ConvBlock3d(in_channels, init_features)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3d(init_features, init_features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock3d(init_features * 2, init_features * 4)
        self.up2 = nn.ConvTranspose3d(
            init_features * 4, init_features * 2, kernel_size=2, stride=2
        )
        self.dec2 = ConvBlock3d(init_features * 4, init_features * 2)
        self.up1 = nn.ConvTranspose3d(
            init_features * 2, init_features, kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock3d(init_features * 2, init_features)
        self.final = nn.Conv3d(init_features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.dec2(torch.cat((self.up2(bottleneck), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.up1(dec2), enc1), dim=1))
        return self.final(dec1)


def center_crop_or_pad(
    volume: np.ndarray, target_shape: Sequence[int]
) -> np.ndarray:
    """Center crop/pad a 3D volume without changing its orientation."""
    if volume.ndim != 3 or len(target_shape) != 3:
        raise ValueError("Expected a 3D volume and a three-dimensional target shape.")
    slices = []
    pad_width = []
    for size, target in zip(volume.shape, target_shape):
        if target < 1:
            raise ValueError("Target dimensions must be positive.")
        if size > target:
            start = (size - target) // 2
            slices.append(slice(start, start + target))
            pad_width.append((0, 0))
        else:
            slices.append(slice(0, size))
            missing = target - size
            before = missing // 2
            pad_width.append((before, missing - before))
    cropped = volume[tuple(slices)]
    if any(before or after for before, after in pad_width):
        cropped = np.pad(cropped, pad_width, mode="constant")
    return np.ascontiguousarray(cropped, dtype=np.float32)


class BrainTumorDataset(Dataset):
    """Read aligned four-modality NIfTI cases from a local CSV manifest.

    Required columns: flair,t1,t1gd,t2,seg. Relative paths are resolved
    relative to the manifest file. Subject identifiers, if present, are ignored.
    """

    def __init__(
        self, csv_path: Path, target_size: Sequence[int] = (96, 96, 96)
    ) -> None:
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.target_size = tuple(int(value) for value in target_size)
        if len(self.target_size) != 3 or any(value < 1 for value in self.target_size):
            raise ValueError("target_size must contain three positive dimensions.")

        with self.csv_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError("Manifest must contain a CSV header.")
            missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
            if missing:
                raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
            self.rows = list(reader)
        if not self.rows:
            raise ValueError("Manifest contains no cases.")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, value: str, row_number: int, column: str) -> Path:
        if not value.strip():
            raise ValueError(f"Empty path in manifest row {row_number}, column {column}.")
        path = Path(value.strip()).expanduser()
        if not path.is_absolute():
            path = self.csv_path.parent / path
        if not path.is_file():
            raise FileNotFoundError(
                f"Input file missing for manifest row {row_number}, column {column}."
            )
        return path

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        row_number = index + 2
        images: list[np.ndarray] = []
        reference_shape = None
        reference_affine = None

        for column in MODALITIES:
            path = self._resolve_path(row[column], row_number, column)
            image = nib.load(str(path))
            volume = np.asarray(image.dataobj, dtype=np.float32)
            if volume.ndim != 3:
                raise ValueError(f"Expected a 3D image in manifest row {row_number}.")
            if not np.isfinite(volume).all():
                raise ValueError(f"Non-finite image values in manifest row {row_number}.")
            if reference_shape is None:
                reference_shape = volume.shape
                reference_affine = image.affine
            elif volume.shape != reference_shape or not np.allclose(
                image.affine, reference_affine, atol=1e-3
            ):
                raise ValueError(
                    f"Modalities are not aligned in manifest row {row_number}."
                )

            volume = center_crop_or_pad(volume, self.target_size)
            mean = float(volume.mean())
            std = float(volume.std())
            volume = (volume - mean) / max(std, 1e-8)
            images.append(volume)

        seg_path = self._resolve_path(row["seg"], row_number, "seg")
        seg_image = nib.load(str(seg_path))
        segmentation = np.asarray(seg_image.dataobj, dtype=np.float32)
        if (
            segmentation.shape != reference_shape
            or not np.allclose(seg_image.affine, reference_affine, atol=1e-3)
        ):
            raise ValueError(
                f"Segmentation is not aligned with MRI in manifest row {row_number}."
            )
        mask = center_crop_or_pad(
            (segmentation > 0).astype(np.float32), self.target_size
        )
        image_tensor = torch.from_numpy(np.stack(images)).float()
        mask_tensor = torch.from_numpy(mask[None]).float()
        return image_tensor, mask_tensor


def dice_iou_metrics(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6
) -> tuple[float, float]:
    prediction = (torch.sigmoid(logits) > threshold).float()
    truth = (target > 0.5).float()
    prediction = prediction.reshape(-1)
    truth = truth.reshape(-1)
    intersection = float((prediction * truth).sum().item())
    predicted_count = float(prediction.sum().item())
    target_count = float(truth.sum().item())
    union = predicted_count + target_count - intersection
    dice = 1.0 if predicted_count + target_count == 0 else (
        2.0 * intersection + eps
    ) / (predicted_count + target_count + eps)
    iou = 1.0 if union == 0 else (intersection + eps) / (union + eps)
    return dice, iou


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    epsilon: float,
    criterion: nn.Module,
) -> torch.Tensor:
    """Create an FGSM perturbation without accumulating model parameter gradients."""
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")
    source = images.detach().requires_grad_(True)
    with torch.enable_grad():
        logits = model(source)
        loss = criterion(logits, masks)
        gradient = torch.autograd.grad(loss, source, only_inputs=True)[0]
        perturbed = source.detach() + epsilon * gradient.sign()
    # Match the supplied experiment's clipping to each input sample's original range.
    lower = images.amin()
    upper = images.amax()
    return perturbed.clamp(min=lower, max=upper).detach()


def evaluate_robustness(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epsilons: Sequence[float],
    threshold: float,
) -> list[dict[str, float | int]]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    results: list[dict[str, float | int]] = []

    for epsilon in epsilons:
        dice_scores: list[float] = []
        iou_scores: list[float] = []
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            attacked = fgsm_attack(model, images, masks, float(epsilon), criterion)
            with torch.no_grad():
                logits = model(attacked)
            dice, iou = dice_iou_metrics(logits, masks, threshold=threshold)
            dice_scores.append(dice)
            iou_scores.append(iou)

        results.append(
            {
                "epsilon": float(epsilon),
                "n_cases": len(dice_scores),
                "mean_dice": float(np.mean(dice_scores)),
                "std_dice": float(np.std(dice_scores)),
                "mean_iou": float(np.mean(iou_scores)),
                "std_iou": float(np.std(iou_scores)),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/fgsm_results.csv"))
    parser.add_argument(
        "--epsilons", type=float, nargs="+", default=[0.0, 0.4, 0.8, 1.0, 2.0]
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--volume-size", type=int, nargs=3, default=(96, 96, 96),
        metavar=("DEPTH", "HEIGHT", "WIDTH"),
    )
    args = parser.parse_args()
    if not args.epsilons or any(value < 0 for value in args.epsilons):
        parser.error("--epsilons must contain non-negative values.")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")
    if any(value < 1 or value % 4 != 0 for value in args.volume_size):
        parser.error("each --volume-size dimension must be a positive multiple of 4.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D().to(device)
    checkpoint = torch.load(
        args.checkpoint.expanduser(), map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    dataset = BrainTumorDataset(args.manifest, args.volume_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = evaluate_robustness(
        model, loader, device, args.epsilons, args.threshold
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"Evaluated {len(dataset)} cases across {len(args.epsilons)} epsilon values.")
    print(f"Aggregate metrics saved to {args.output}. No case identifiers were written.")


if __name__ == "__main__":
    main()
