# TEKNOFEST 2026 - Reference Object Matching

Reference Object Matching benchmark and matching engine developed for the TEKNOFEST 2026 Autonomous UAV Competition.

> **Current Status:** 🚧 Under Development

---

# Project Goal

The objective of this project is to detect and localize a given reference object inside aerial video frames.

Unlike traditional object detection, the system receives one or more reference images at runtime and must locate the corresponding object in unseen video frames.

The solution is designed to be:

- Dataset independent
- Algorithm independent
- Modular
- Easily extensible
- Benchmark-driven

---

# Current Architecture

```
Validation Dataset
        │
        ▼
Dataset Loader
        │
        ▼
CVAT XML Parser
        │
        ▼
Benchmark Runner
        │
        ▼
Matching Engine
        │
        ▼
Predictions
```

The benchmark infrastructure is completely separated from the matching algorithms.

Any future matching algorithm can be integrated by implementing the `MatchingEngine` interface.

---

# Project Structure

```
Project/
│
├── benchmark/
│   ├── data_models.py
│   ├── loader.py
│   ├── parsers.py
│   ├── prediction_models.py
│   ├── engine.py
│   ├── runner.py
│   └── engines/
│
├── Validation/          # Ignored by Git
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Features

- Modular benchmark architecture
- CVAT XML 1.1 parser
- Validation dataset loader
- Matching Engine abstraction
- Strategy Pattern architecture
- Dependency Injection
- Dummy engine for pipeline validation

---

# Planned Features

- DINOv2 global descriptors
- Local feature extraction
- Geometric verification
- Confidence estimation
- IoU evaluation
- Precision / Recall
- mAP metrics
- Runtime benchmarking

---

# Design Principles

The project follows modern software engineering principles:

- SOLID
- Strategy Pattern
- Dependency Injection
- Separation of Concerns
- Extensible architecture

The benchmark is completely independent from the matching algorithms.

---

# Validation Dataset

Each validation set contains:

- Reference image
- Test images
- Ground-truth annotations (CVAT XML)

```
Validation/
│
├── ref01/
├── ref02/
├── ref03/
├── ref04/
├── ref05/
└── ref06/
```

The validation dataset is intentionally excluded from the repository.

---

# Getting Started

Clone the repository:

```bash
git clone <repository-url>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run benchmark (coming soon):

```bash
python run_benchmark.py
```

---

# Roadmap

- [x] Benchmark architecture
- [x] Dataset loader
- [x] XML parser
- [x] Matching Engine interface
- [x] Dummy engine
- [ ] First real Matching Engine
- [ ] DINOv2 integration
- [ ] Geometric verification
- [ ] Evaluation metrics
- [ ] Performance optimization

---

# Technologies

- Python
- OpenCV
- NumPy
- PyTorch *(planned)*
- DINOv2 *(planned)*
- LightGlue *(planned)*

---

# License

This repository is currently intended for the TEKNOFEST 2026 competition.

```