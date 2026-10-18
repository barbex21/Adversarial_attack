# Reported Results

This page records aggregate values transcribed from the project's supplied thesis results document. They have not been independently reproduced in this repository.

## Recorded values

| Measure | Reported value | Context |
| --- | ---: | --- |
| Average test Dice | 0.8230 (82.30%) | Recorded under the final-results section |
| Training loss | 0.0356 | Epoch 22, last epoch shown in the supplied log |
| Validation loss | 0.0588 | Epoch 22, last epoch shown in the supplied log |

These values describe the reported segmentation run. They are not aggregate FGSM robustness results and should not be interpreted as such. The source also contains example visualizations; they are omitted here, and no MRI scans, segmentation masks, patient-level records, or source document are included in this repository.

## Reproducibility status

The values above are transcribed from the supplied results document and have not been recalculated from predictions in this repository. The repository's FGSM evaluator has not been run against the original data or checkpoint. These values therefore do not validate the evaluator or establish model performance under attack.

To reproduce them, the exact data split, preprocessing, checkpoint, threshold, metric aggregation method, and software environment must be available and documented.