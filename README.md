# TEKNOFEST One-Shot Object Matching Benchmark

A modular and extensible benchmark framework for evaluating one-shot object matching algorithms on the TEKNOFEST Autonomous Target Detection dataset.

## Project Overview

This repository is designed around clean software architecture principles, ensuring that new matching algorithms can be benchmarked without modifying the core benchmark pipeline. It supports both RGB and Thermal images using a single, unified pipeline.

## Competition Objective

The goal of this project is to solve the TEKNOFEST One-Shot Object Matching problem. Given a single reference image and a sequence of drone camera frames, the system must detect whether the reference object exists inside each frame and return its bounding box.

## Motivation

The long-term objective is to benchmark different one-shot object matching approaches (like DINOv2 + LightGlue, LoFTR, SuperGlue, OpenCV baselines) under a unified evaluation framework without changing the infrastructure. This enables researchers to rapidly test, evaluate, and optimize new matching models for both RGB and Thermal datasets seamlessly.

## Repository Architecture

The architecture is highly modular, heavily relying on abstractions, the Strategy Pattern, and Dependency Injection.

### Folder Structure

```text
Project/
├── Validation/           # Validation dataset
├── benchmark/            # Core benchmark framework
│   ├── engines/          # Concrete engine implementations
│   └── examples/         # Dummy implementations for testing
├── docs/                 # Comprehensive documentation
│   ├── decisions/        # Architecture Decision Records (ADRs)
│   └── images/           # Architecture diagrams
├── AI_HANDOFF.md         # Shared memory state for AI assistants
├── CHANGELOG.md          # Semantic version history
├── CLAUDE.md             # Project understanding for AI assistants
├── CONTRIBUTING.md       # Guidelines for contributors
├── DEVELOPMENT_GUIDE.md  # Comprehensive development workflow
├── PROJECT_ROADMAP.md    # Milestone roadmap
└── README.md             # This file
```

## Current Implementation Status

**Completed:**
- Validation dataset loader & CVAT XML parser
- Dataset data models & Prediction models
- Benchmark runner & Metrics interface
- MatchingEngine abstraction
- Coarse-to-Fine engine architecture
- FeatureExtractor, CoarseLocalizer, FineMatcher, GeometricVerifier abstractions
- Dummy engine for pipeline verification
- DINOv2 FeatureExtractor concrete class (RGB & Thermal support)
- CosineHeatmapLocalizer (Global Similarity & Coarse Localization)
- AlikedLightGlueMatcher (Fine Matching with ALIKED + LightGlue)
- RansacHomographyVerifier (Geometric Verification with OpenCV RANSAC)

**In Progress / Next Task:**
- Benchmark metrics integration (IoU, precision, recall) (Phase 7)

## Pipeline Overview

The targeted Coarse-to-Fine pipeline runs sequentially:
1. **Reference Image & Test Image**
2. **DINOv2 Feature Extraction** (Global Similarity & Candidate Localization)
3. **ALIKED Keypoint Detection**
4. **LightGlue Matching**
5. **Geometric Verification (RANSAC)**
6. **Bounding Box Projection & Evaluation Metrics**

## RGB + Thermal Philosophy

The pipeline is explicitly designed to support BOTH RGB and Thermal images using ONE unified pipeline. Thermal images are converted into compatible network input during preprocessing. Separate models for RGB and Thermal are explicitly NOT planned unless future benchmark results prove an insurmountable performance gap.

## Installation

*(Installation instructions to be added)*

## Quick Start / How to Run the Benchmark

*(Execution scripts to be added)*

## Repository Philosophy

- **Strategy Pattern & Dependency Injection**: High cohesion, low coupling.
- **Single Responsibility Principle**: Each component does one thing well.
- **Immutable Data Models**: Interfaces use frozen dataclasses.
- **Framework-independent Interfaces**: The benchmark core doesn't depend on PyTorch, OpenCV, etc.

## Future Roadmap & Future Work

See [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) for detailed milestones, which include implementing DINOv2 Feature Extractor, ALIKED, LightGlue, metrics, optimizations for Jetson, and final Teknofest Submission. 
See [docs/future_work.md](docs/future_work.md) for a prioritized implementation backlog.