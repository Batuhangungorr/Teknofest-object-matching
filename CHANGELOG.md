# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Upcoming Work
- Add IoU / Precision / Recall metrics.
- End-to-end benchmark evaluation.

## [0.1.0] - 2026-06-27

### Completed Work
- **Added** Validation dataset loader.
- **Added** CVAT XML parser.
- **Added** Dataset data models (using frozen dataclasses).
- **Added** Benchmark runner (orchestrator).
- **Added** Prediction models for metrics.
- **Added** `MatchingEngine` abstraction.
- **Added** Metrics interface skeleton.
- **Added** `CoarseToFineEngine` architecture skeleton.
- **Added** `FeatureExtractor` abstraction.
- **Added** `CoarseLocalizer` abstraction.
- **Added** `FineMatcher` abstraction.
- **Added** `GeometricVerifier` abstraction.
- **Added** `DINOv2FeatureExtractor` concrete class implementation.
- **Added** `CosineHeatmapLocalizer` concrete class implementation.
- **Added** `AlikedLightGlueMatcher` concrete class implementation (Kornia).
- **Added** `RansacHomographyVerifier` concrete class implementation (OpenCV).
- **Added** Dummy engine for pipeline verification.
- **Changed** Reorganized documentation to standard structure.
