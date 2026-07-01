# ADR 0006: Modular Engine Design

**Status:** Accepted
**Date:** 2026-06-27

## Context
A complete one-shot object matching pipeline involves several distinct mathematical and deep learning steps: feature extraction, localization, fine matching, and geometric verification. 

## Decision
We decided to adopt a modular `CoarseToFineEngine` that delegates work to specific, single-responsibility abstract components (`FeatureExtractor`, `CoarseLocalizer`, `FineMatcher`, `GeometricVerifier`).

## Alternatives
- **Monolithic Engine**: A single `DinoLightGlueEngine` class that handles everything internally from pixels to bounding box.
- **End-to-End Network**: A single neural network forward pass that outputs bounding boxes directly.

## Consequences
- **Positive**: 
  - We can ablate specific components (e.g., keep ALIKED+LightGlue but swap DINOv2 for another feature extractor).
  - Researchers can work on different parts of the pipeline simultaneously without merge conflicts.
  - Easier to write isolated unit tests for the RANSAC logic, for instance.
- **Negative**:
  - Increased complexity in orchestrating data between modules.
  - Data transfer overhead between modules compared to a single fused forward pass.

## Future Implications
The `CoarseToFineEngine` acts merely as a data pipeline. Future performance optimizations might require fusing some of these steps (e.g., combining extraction and localization), which might necessitate creating a new class implementing the top-level `MatchingEngine` interface rather than reusing the `CoarseToFineEngine`.
