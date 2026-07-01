# Design Decisions

This document logs all major architectural design decisions.

## Strategy Pattern
- **Context**: The benchmark requires testing many different combinations of extractors, localizers, and matchers.
- **Decision**: Define algorithms using the Strategy Pattern via abstract base classes (e.g., `MatchingEngine`).
- **Motivation**: Allows hot-swapping algorithms at runtime without modifying the benchmark runner.
- **Advantages**: Extremely extensible; adheres to Open/Closed Principle.
- **Disadvantages**: Introduces abstraction overhead.
- **Long-term impact**: Facilitates rapid integration of new state-of-the-art models as they are published.

## Dependency Injection
- **Decision**: Components receive their dependencies (like specific FeatureExtractors) through constructors.
- **Motivation**: Decouples the creation of objects from their usage, making the `CoarseToFineEngine` highly testable and flexible.

## Immutable Dataclasses
- **Decision**: All data transferred between modules (e.g., `DetectionPrediction`) uses Python's `@dataclass(frozen=True)`.
- **Motivation**: Prevents accidental state mutation during the pipeline execution.

## Abstract Interfaces & Modular Benchmark
- **Decision**: The core `benchmark/` folder contains only interfaces and orchestration logic.
- **Motivation**: Separates the "what" (benchmarking) from the "how" (neural networks).

## Framework-Independent Interfaces
- **Decision**: PyTorch, TensorFlow, OpenCV, etc., cannot be imported in the abstract base classes.
- **Motivation**: Ensures the benchmark isn't locked into one ML ecosystem.

## Unified RGB + Thermal Pipeline
- **Decision**: Both modalities use the exact same matching pipeline and weights.
- **Motivation**: Simplifies the architecture. Relying on the zero-shot structural generalization of models like DINOv2 allows testing without maintaining two separate codebases or training regimes.
