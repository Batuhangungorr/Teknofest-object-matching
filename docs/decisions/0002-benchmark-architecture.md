# ADR 0002: Benchmark Architecture

**Status:** Accepted
**Date:** 2026-06-27

## Context
To evaluate matching algorithms, we need a mechanism that feeds data into the algorithms, records predictions, and computes metrics. We must ensure the evaluation logic doesn't become entangled with the neural network logic.

## Decision
We decided to strictly separate orchestration (`BenchmarkRunner`) from algorithmic logic (`MatchingEngine`). The `BenchmarkRunner` only knows about the abstract `MatchingEngine` interface. It passes reference/test pairs to the engine and collects `DetectionPrediction` objects. A separate `MetricsCalculator` evaluates these predictions against the ground truth.

## Alternatives
- **Integrated Evaluator**: Having the neural network code load the dataset and compute its own metrics. Rejected because it duplicates evaluation logic for every new algorithm tested and introduces bias.

## Consequences
- **Positive**: Guarantees fair, standardized evaluation across all algorithms. Simplifies testing.
- **Negative**: Requires strict adherence to data contracts (`DetectionPrediction`).

## Future Implications
We can write a new algorithm completely independently and, as long as it fulfills the `MatchingEngine` contract, instantly benchmark it against all historical models.
