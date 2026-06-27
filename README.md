# TEKNOFEST One-Shot Object Matching Benchmark

A modular and extensible benchmark framework for evaluating one-shot object matching algorithms on the TEKNOFEST Autonomous Target Detection dataset.

The project is designed around clean software architecture principles so that new matching algorithms can be benchmarked without modifying the benchmark pipeline.

---

# Current Status

## Completed

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
- Dummy engine for pipeline verification

## In Progress

- GeometricVerifier abstraction

## Planned

- DINOv2 Feature Extractor
- Coarse Heatmap Localization
- ALIKED + LightGlue Matcher
- OpenCV RANSAC Verification
- IoU / Precision / Recall metrics
- End-to-end benchmark evaluation

---

# Project Structure

```
Project/
│
├── Validation/
│
└── benchmark/
    ├── data_models.py
    ├── prediction_models.py
    ├── loader.py
    ├── parsers.py
    ├── runner.py
    ├── engine.py
    ├── metrics.py
    │
    ├── engines/
    │   └── coarse_to_fine/
    │       ├── feature_extractor.py
    │       ├── localizer.py
    │       ├── matcher.py
    │       ├── verifier.py
    │       └── engine.py
    │
    └── examples/
        ├── dummy_engine.py
        └── run_dummy_benchmark.py
```

---

# Architecture

```
Validation Dataset
        │
        ▼
Dataset Loader
        │
        ▼
BenchmarkRunner
        │
        ▼
MatchingEngine
        │
        ▼
CoarseToFineEngine
        │
 ┌──────┼────────────┬────────────┐
 ▼      ▼            ▼            ▼
Feature Localizer  Matcher    Verifier
Extractor
```

---

# Design Principles

- Strategy Pattern
- Dependency Injection
- Single Responsibility Principle
- Open/Closed Principle
- Framework-independent interfaces
- Algorithm-independent benchmark pipeline

---

# Goal

The long-term objective is to benchmark different one-shot object matching approaches under a unified evaluation framework.

Planned implementations include:

- DINOv2 + LightGlue
- LoFTR
- SuperGlue
- OpenCV-based baselines
- Future transformer-based matching models

without changing the benchmark infrastructure.

---

# License

MIT