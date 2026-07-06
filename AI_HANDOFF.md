# AI Handoff

This file is the permanent shared memory between ChatGPT, Claude, Gemini, and future AI assistants to continue development seamlessly.

## Current Stage

Architecture abstraction is mostly complete. Implementation phase is starting.

## Current Architecture

The architecture is built around a `BenchmarkRunner` that orchestrates `MatchingEngine` implementations. The targeted pipeline is a `CoarseToFineEngine` which utilizes abstractions for `FeatureExtractor`, `CoarseLocalizer`, `FineMatcher`, and `GeometricVerifier`.

## Completed Modules

- Validation dataset loader
- CVAT XML parser
- Dataset data models
- Benchmark runner
- Prediction models
- MatchingEngine abstraction
- Metrics interface
- Coarse-to-Fine engine architecture
- FeatureExtractor abstraction
- CoarseLocalizer abstraction
- FineMatcher abstraction
- GeometricVerifier abstraction
- Dummy engine for pipeline verification
- DINOv2 FeatureExtractor concrete class
- CosineHeatmapLocalizer (Global Similarity & Coarse Localization)

- AlikedLightGlueMatcher (Fine Matching with ALIKED + LightGlue)
- RansacHomographyVerifier (Geometric Verification with OpenCV RANSAC)
- MetricsCalculator (IoU, precision, recall)
- Evaluation Script (RGB + Thermal Validation)

## Modules in Progress

- Performance Optimization (ONNX / TensorRT)

## Next Implementation Target

Implement Performance Optimization (Phase 9).
**Constraint**: Convert the models to ONNX or TensorRT to reduce latency and increase throughput for real-time constraints.

## Current Benchmark State

Dummy benchmark exists for verifying the pipeline flow. No real algorithms have been benchmarked yet.

## Important Architectural Decisions

- **Single Pipeline**: Both RGB and Thermal modalities are supported through a single unified pipeline. Separate pipelines are not allowed.
- **Framework Independence**: Core abstractions do not depend on ML frameworks.
- **Strict Separation**: `BenchmarkRunner` only orchestrates; it does not match or evaluate.

## Current Branch

main (assumed)

## Current Version

v0.1.0 (architecture phase)

## Latest Commit

[Requires Git Context]

## Known Limitations

- No concrete models integrated yet.
- Metrics evaluation is only a skeleton.

## Known Issues

None currently reported.

## Future Milestones

- DINOv2 Feature Extractor
- Global Similarity
- Heatmap Generation
- Candidate Localization
- ALIKED Local Feature Detector
- LightGlue Feature Matcher
- OpenCV RANSAC Geometric Verification
- IoU / Precision / Recall Metrics
- Jetson Optimization