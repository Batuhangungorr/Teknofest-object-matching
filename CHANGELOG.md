# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Upcoming Work
- Implement GeometricVerifier abstraction.
- Integrate DINOv2 Feature Extractor.
- Implement Coarse Heatmap Localization.
- Integrate ALIKED + LightGlue Matcher.
- Implement OpenCV RANSAC Verification.
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
- **Added** Dummy engine for pipeline verification.
- **Changed** Reorganized documentation to standard structure.
