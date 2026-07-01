# Contributing Guide

Welcome to the TEKNOFEST One-Shot Object Matching Benchmark repository. We value your contributions. This guide details the rules and workflows for contributing.

## Coding Standards

- Follow PEP 8 guidelines for Python.
- Provide type hints for all public methods and functions.
- Every public class and method must include comprehensive docstrings.
- Use immutable frozen dataclasses for data transfer objects.
- Rely on Dependency Injection and the Strategy Pattern.
- Keep framework imports (like PyTorch or OpenCV) strictly inside concrete implementation files. Do not leak them into abstract base classes.

## Documentation Standards

- Always synchronize documentation with code changes.
- Never duplicate documentation. Update the single source of truth in `docs/`.
- Ensure all markdown is professionally formatted and internally consistent.

## Git Workflow

1. **Branch Naming**: 
   - `feature/<name>` for new features.
   - `fix/<name>` for bug fixes.
   - `docs/<name>` for documentation updates.
2. **Commit Naming**:
   - Small, logical commits are preferred. One logical task per commit.
   - Use conventional commits format (e.g., `feat: add DINOv2 extractor`, `docs: update AI_HANDOFF`).
3. **Pull Requests**:
   - Keep pull requests focused on a single responsibility.
   - Provide clear descriptions of the changes.

## Review Workflow

- All pull requests require review by a maintainer before merging.
- Architecture changes require rigorous review and an accompanying ADR in `docs/decisions/`.

## Verification Requirements

- Every new implementation must pass the existing benchmark tests (e.g., `examples/run_dummy_benchmark.py`).
- Do not break backward compatibility.
- Ensure the pipeline can still process both RGB and Thermal datasets smoothly without modifications.
