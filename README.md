# TEKNOFEST One-Shot Object Matching Benchmark

A modular benchmark framework for evaluating one-shot object matching algorithms on the TEKNOFEST Object Detection Challenge validation dataset.

The project is designed so that **new matching algorithms can be added without modifying the benchmark infrastructure**.

---

# Project Goals

This project aims to provide a clean and extensible benchmark for evaluating different one-shot object matching approaches such as:

- DINOv2
- LightGlue
- LoFTR
- SuperGlue
- OpenCV-based methods
- Future custom algorithms

The benchmark infrastructure is completely independent from any specific algorithm.

---

# Features

- Modular architecture
- Strategy Pattern based engine system
- Dependency Injection
- CVAT XML parser
- Automatic Validation dataset loader
- Strongly typed data models
- Prediction data models
- Benchmark runner
- Extensible engine architecture
- Future-ready metrics module
- Example DummyEngine for pipeline testing

---

# Project Structure

```text
Project/
│
├── Validation/
│   ├── ref01/
│   ├── ref02/
│   ├── ...
│   └── ref06/
│
└── benchmark/
    ├── __init__.py
    ├── data_models.py
    ├── prediction_models.py
    ├── loader.py
    ├── parsers.py
    ├── engine.py
    ├── runner.py
    ├── metrics.py
    │
    ├── engines/
    │   └── coarse_to_fine/
    │       ├── engine.py
    │       ├── feature_extractor.py
    │       ├── localizer.py
    │       ├── matcher.py
    │       └── verifier.py
    │
    └── examples/
        ├── dummy_engine.py
        └── run_dummy_benchmark.py
```

---

# Architecture

The benchmark follows the Strategy Pattern.

```
BenchmarkRunner
        │
        ▼
 MatchingEngine (ABC)
        ▲
        │
 ┌──────┼────────────┐
 │      │            │
 │      │            │
DINOv2  LightGlue  LoFTR
 │
Custom Engines
```

The benchmark never depends on a concrete algorithm.

It only communicates with the `MatchingEngine` interface.

---

# Pipeline

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
DetectionPrediction
        │
        ▼
Metrics (future)
```

---

# Current Components

## Dataset Loader

Automatically scans

```
Validation/
    ref01/
    ref02/
    ...
```

and loads

- reference image
- Images folder
- annotations.xml

into Python objects.

---

## Parser

Reads CVAT XML annotations and converts them into

- BoundingBox
- ImageAnnotation

objects.

---

## MatchingEngine

Abstract interface implemented by every algorithm.

Required methods:

```python
initialize()

set_reference()

detect()

cleanup()
```

Future algorithms only need to implement this interface.

---

## BenchmarkRunner

Responsible for

- initializing engine
- iterating over reference sets
- running inference
- collecting predictions

It never performs object matching itself.

---

## Prediction Models

Ground truth and predictions are intentionally separated.

Ground Truth

```
BoundingBox
ImageAnnotation
ReferenceSet
```

Predictions

```
DetectionPrediction
SetPredictions
BenchmarkPredictions
```

This makes future metric computation much cleaner.

---

## Metrics

A placeholder module reserved for future evaluation metrics.

Planned metrics include:

- IoU
- Precision
- Recall
- mAP
- Success Rate
- Average Processing Time

---

# Example

```python
from benchmark.loader import load_validation_dataset
from benchmark.runner import BenchmarkRunner
from benchmark.examples.dummy_engine import DummyEngine

dataset = load_validation_dataset("Validation")

engine = DummyEngine()

runner = BenchmarkRunner(
    engine=engine,
    dataset=dataset
)

results = runner.run()
```

---

# Planned Engine

The first real implementation will be

```
Coarse-to-Fine Engine
```

Pipeline:

```
Reference Image
      │
      ▼
Feature Extractor (DINOv2)
      │
      ▼
Coarse Localizer
      │
      ▼
Candidate Region
      │
      ▼
LightGlue Matcher
      │
      ▼
Geometric Verification
      │
      ▼
Bounding Box
```

---

# Design Principles

- Strategy Pattern
- Dependency Injection
- SOLID Principles
- Single Responsibility Principle
- Open/Closed Principle
- Type-safe Data Models
- Algorithm-independent Benchmark

---

# Current Status

✅ Dataset Loader

✅ XML Parser

✅ Data Models

✅ Prediction Models

✅ MatchingEngine Interface

✅ BenchmarkRunner

✅ DummyEngine

✅ Benchmark Pipeline

✅ Coarse-to-Fine Skeleton

⬜ DINOv2 Feature Extraction

⬜ Coarse Localization

⬜ LightGlue Matching

⬜ Geometric Verification

⬜ Metrics Implementation

---

# Future Roadmap

- DINOv2 integration
- LightGlue integration
- TensorRT optimization
- Jetson deployment
- IoU & mAP evaluation
- Visualization utilities
- Benchmark report generation
- Multi-engine comparison
- CLI support

---

# License

This repository is intended for educational and research purposes.