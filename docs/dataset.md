# Dataset

This document explains the dataset structures and assumptions used in the benchmark framework.

## Validation Dataset

The framework operates on the TEKNOFEST Autonomous Target Detection dataset. The dataset consists of sequences of drone camera frames and isolated reference images.

## Reference Sets and Ground Truth

- **Reference Images**: Isolated shots of the object to be detected.
- **Test Frames**: Full camera frames where the object may or may not be present.
- **Ground Truth**: Annotations are provided via CVAT XML files, containing bounding boxes for the targets in the test frames. The XML parser converts these into internal immutable data models.

## Current Dataset Layout

Currently, the `Validation/` directory holds the test sets, including the XML annotations and images. The loader component iterates through these structures to yield reference/test pairs.

## Future RGB + Thermal Organization

The benchmark supports BOTH RGB and Thermal datasets. 
- **Dataset Philosophy**: The dataset loader will handle both modalities transparently.
- **Single Pipeline**: The matching pipeline itself will not branch based on modality. Thermal images will be preprocessed to be compatible with models trained on RGB (e.g., standardizing channels). This ensures we can evaluate the exact same algorithmic pipeline on different sensor modalities.

## Dataset Assumptions

- Each query consists of exactly one reference image and one test frame.
- The ground truth is assumed to be accurate bounding boxes.
- Missing ground truth implies the target is absent from the frame.
