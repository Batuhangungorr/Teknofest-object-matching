# Future Work

This document serves as the prioritized implementation backlog for the project.

## Remaining Modules (High Priority)
1. **GeometricVerifier Abstraction**: Finalize interface.
2. **DINOv2 Feature Extractor**: Wrap DINOv2 to output dense features.
3. **Coarse Heatmap Localizer**: Convert global features into bounding box candidates.
4. **ALIKED & LightGlue Integration**: Implement the FineMatcher.
5. **RANSAC Geometric Verification**: Implement OpenCV based homography/fundamental matrix estimation.
6. **Metrics Integration**: Implement IoU, Precision, Recall calculations.

## Research Tasks (Medium Priority)
- Investigate the impact of various thermal preprocessing techniques (e.g., histogram equalization, pseudocolor) on DINOv2 zero-shot performance.
- Evaluate LoFTR vs. LightGlue in the FineMatcher stage on thermal sequences.

## Optimization Tasks
- Profile the `CoarseToFineEngine` latency.
- Export DINOv2 and LightGlue to ONNX.
- Apply INT8 quantization and TensorRT calibration for the Jetson platform.

## Deployment Tasks
- Develop a TCP/UDP client for live video stream ingestion.
- Implement JSON serializers for bounding box telemetry output to ground stations.
- Containerize the application for Jetson (Docker + NVIDIA Runtime).

## Documentation Tasks
- Document the internal tensors shapes (e.g., `(B, C, H, W)`) throughout the pipeline.
- Produce video/GIF demonstrations of the pipeline working on validation sets.

## Long-term Improvements
- Support multi-object tracking (MOT) across frames using temporal heuristics.
- Investigate end-to-end differentiable matching architectures.
