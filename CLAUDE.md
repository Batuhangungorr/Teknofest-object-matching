# AI Assistant Guide (CLAUDE.md)

This document exists ONLY for AI assistants (Claude, Gemini, ChatGPT). It provides the critical context and rules necessary to maintain the integrity of this repository. Future AI assistants must understand the project after reading ONLY this file.

## Project Philosophy

This repository is a research-grade, production-quality open-source benchmark framework for the TEKNOFEST One-Shot Object Matching competition. Architecture stability and modularity are prioritized over implementation speed. The benchmark supports BOTH RGB and Thermal images using ONE unified pipeline. Separate models for RGB and Thermal are explicitly NOT planned.

## Architecture Philosophy

- **Strategy Pattern**: Concrete implementations must inherit from abstract base classes.
- **Dependency Injection**: Dependencies are passed into constructors.
- **Abstract Interfaces**: The core benchmark (`benchmark/`) operates solely on interfaces.
- **Immutable Data Contracts**: Use frozen dataclasses for data transfer objects.
- **Framework Independence**: The core interfaces must not import PyTorch, TensorFlow, OpenCV, etc. These belong only in concrete implementations.

## Development Philosophy

- **Modular Design**: Every component must have a single responsibility.
- **One prompt = one logical implementation**: Do not attempt to refactor the entire codebase in one go.
- **Architecture over Speed**: Do not redesign the architecture unless explicitly requested by the user.

## Coding Philosophy

- **Type Hints**: All public methods must contain type hints.
- **Docstrings**: Every public class and method must include detailed docstrings.
- **Immutability**: Use frozen dataclasses whenever possible.
- **Composition over Inheritance**: Prefer composing behaviors rather than deep class hierarchies.

## Naming Conventions

- Use explicit, descriptive names for classes and methods.
- Follow PEP 8 guidelines for Python code.

## Repository Conventions

- There must NEVER be duplicate documentation. Merge useful information and delete the obsolete file.
- Use lowercase filenames everywhere inside the `docs/` directory.

## Documentation Conventions

- Documentation must be internally consistent.
- `docs/decisions/` contains Architecture Decision Records (ADRs).
- `README.md` is the primary entry point.
- `AI_HANDOFF.md` serves as shared memory for AI continuity.

## Verification Workflow

- Always run verification and check existing benchmark tests before concluding a task.
- Ensure 0 changes were made to the core implementation or `benchmark/` if the task is documentation-focused.

## Prompt Workflow

- Read `AI_HANDOFF.md` to understand the current stage.
- Follow the exact constraints provided by the user.
- Complete the single requested task.
- Update `AI_HANDOFF.md` with the new state.

## Forbidden Modifications

- Do NOT modify public APIs unless explicitly requested.
- Do NOT modify `benchmark/` or `Validation/` directories or existing data models unless necessary.
- Do NOT modify any Python source file if the task is just documentation.
- `BenchmarkRunner` is for orchestration only. It must never load neural networks, perform matching, or compute metrics.
- `MetricsCalculator` must remain independent from matching algorithms.

## Expected Implementation Order

1. DINOv2 Feature Extractor
2. Global Similarity
3. Heatmap Generation
4. Candidate Localization
5. ALIKED
6. LightGlue
7. Geometric Verification (RANSAC)
8. Metrics
9. RGB + Thermal Validation
10. Jetson Optimization
11. Teknofest Submission

## Expected Reasoning Style

Think critically about architectural implications before writing code. Prioritize the separation of concerns. If unsure about a design decision, consult the existing abstractions and follow the established pattern rather than inventing a new one. Do not break backward compatibility.
