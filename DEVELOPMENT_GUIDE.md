# Development Guide

This guide explains the complete development workflow for contributing to the TEKNOFEST One-Shot Object Matching Benchmark repository. All developers and AI assistants must follow these rules.

## Development Philosophy

The primary objective is to maintain a research-grade, production-quality framework. Architecture stability, modularity, and clean abstractions are prioritized over rapid, monolithic feature development. 

## Core Rules

1. **One prompt = one logical implementation**: Do not attempt to do multiple unrelated tasks in a single commit or prompt. Keep changes scoped.
2. **One implementation file per prompt**: Focus modifications on the specific component being developed.
3. **Never modify unrelated files**: Prevent scope creep. If a task requires modifying `localizer.py`, do not refactor `metrics.py`.
4. **Never refactor architecture unless requested**: The abstractions and interfaces are deliberately designed. Implement within the established boundaries.
5. **Always preserve interfaces**: Do not change public APIs, function signatures, or data contracts.
6. **Always preserve modularity**: Each component (Extractor, Localizer, Matcher, Verifier) must have a single responsibility.
7. **Always verify imports**: Ensure no framework-specific imports (like `torch` or `cv2`) leak into abstract interfaces.
8. **Always run verification**: Every implementation must pass the existing dummy benchmark tests to ensure the pipeline isn't broken.
9. **Always synchronize documentation**: If an implementation detail is finalized, update the relevant documentation in `docs/` and `AI_HANDOFF.md`.
10. **Always maintain repository consistency**: Ensure there is no duplicated information in the documentation.

## Complete Workflow (From Beginning to End)

1. **Understand the Goal**: Read `README.md` and `AI_HANDOFF.md` to understand the current repository state and the next implementation target.
2. **Review Architecture**: Check `docs/architecture.md` and `docs/pipeline.md` to understand where the new component fits in.
3. **Implement Strategy**: 
    - If adding a new matching algorithm, inherit from the appropriate base class (e.g., `FeatureExtractor`).
    - Use dependency injection to pass required configurations.
    - Keep framework dependencies contained strictly within your concrete implementation class.
4. **Data Models**: Use existing immutable frozen dataclasses for I/O. If a new model is required, ensure it doesn't break backward compatibility.
5. **Type Hints & Documentation**: Ensure all new methods have accurate Python type hints and comprehensive docstrings.
6. **Verify Pipeline**: Run the benchmark suite using the provided scripts (e.g., `examples/run_dummy_benchmark.py`) to confirm orchestration remains intact.
7. **Update State**: Once the code works, update the `AI_HANDOFF.md` to reflect the completed module and define the next target. Remove obsolete documentation if merging information.
8. **Submit Changes**: Follow the commit and review guidelines in `CONTRIBUTING.md`.
