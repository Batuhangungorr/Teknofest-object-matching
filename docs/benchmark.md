# Benchmark Infrastructure

This document details the core benchmarking components responsible for evaluating algorithms.

## BenchmarkRunner

The `BenchmarkRunner` is the central orchestrator. It receives a dataset loader and an instantiated `MatchingEngine`. It iterates over the dataset, feeds reference and test images into the engine, and collects the resulting `DetectionPrediction` objects.

**Key constraint**: The `BenchmarkRunner` must never know *how* the matching occurs. It only knows the engine's public interface.

## MatchingEngine

The `MatchingEngine` is an abstract base class (Strategy Pattern). It defines the `match()` method which takes a reference image and a test image, and returns a `DetectionPrediction`. 

Current architectures utilize a `CoarseToFineEngine` which implements this interface by composing smaller strategies (FeatureExtractor, CoarseLocalizer, FineMatcher, GeometricVerifier) via Dependency Injection.

## Prediction Models

Prediction models are immutable data contracts (frozen dataclasses) that standardize the output of any matching engine. A standard `DetectionPrediction` contains:
- Bounding box coordinates (x_min, y_min, x_max, y_max)
- Confidence score
- Match validity boolean (True if target found, False otherwise)

## Metrics

The `MetricsCalculator` consumes `DetectionPrediction` objects alongside ground truth data. It evaluates performance regardless of the underlying algorithm. Expected metrics include:
- Intersection over Union (IoU)
- Precision & Recall
- Mean Average Precision (mAP)
- Frames Per Second (FPS) & Latency

## Current Benchmark Architecture

The framework is highly modular. You instantiate concrete implementations of the abstract components and inject them into the engine, which is then injected into the runner:

```python
# Conceptual Example
extractor = DINOv2Extractor()
localizer = HeatmapLocalizer()
matcher = LightGlueMatcher()
verifier = RANSACVerifier()

engine = CoarseToFineEngine(extractor, localizer, matcher, verifier)

runner = BenchmarkRunner(dataset_loader, engine, metrics_calculator)
runner.run()
```
