# ADR 0004: DINOv2 Selection

**Status:** Accepted
**Date:** 2026-06-27

## Context
For the `CoarseLocalizer` stage, we need a method to extract robust semantic features from both the reference image and the large test frame. The features must be robust to extreme scale changes, viewpoint variations, and domain shifts (RGB to Thermal).

## Decision
We selected Meta's DINOv2 (Vision Transformer) as the primary global feature extractor.

## Alternatives
- **ResNet50 / ConvNeXt (ImageNet Pretrained)**: Fast, but supervised features are less robust to domain shifts and out-of-distribution targets compared to self-supervised ViT features.
- **CLIP**: Good for zero-shot text-to-image, but local patch-level features are less spatially consistent than DINOv2.

## Consequences
- **Positive**: State-of-the-art semantic correspondence without any fine-tuning. Excellent structural representation that generalizes to thermal images.
- **Negative**: High computational cost. ViTs are slower and consume more memory than CNNs, necessitating a coarse-to-fine approach rather than running dense matching directly on high-resolution outputs.

## Future Implications
The pipeline relies on the self-supervised properties of DINOv2. Optimization efforts (like TensorRT) will need to specifically target ViT architectures.
