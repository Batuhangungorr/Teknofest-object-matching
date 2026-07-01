# ADR 0005: LightGlue Selection

**Status:** Accepted
**Date:** 2026-06-27

## Context
Once the `CoarseLocalizer` crops a candidate region, we need to extract and match local keypoints precisely to estimate the final bounding box geometry. This requires a local feature detector and a matcher.

## Decision
We selected ALIKED for keypoint detection and LightGlue for feature matching.

## Alternatives
- **SuperPoint + SuperGlue**: The previous industry standard. LightGlue is fundamentally a faster, more accurate, and more adaptable successor to SuperGlue.
- **LoFTR**: A detector-free matcher. Excellent performance, but highly computationally expensive. Since we are already cropping to a candidate region, extracting keypoints (ALIKED) and matching them (LightGlue) provides a better speed/accuracy tradeoff.

## Consequences
- **Positive**: Extremely fast and accurate fine matching. LightGlue's adaptive depth allows it to stop computation early for easy matches, saving inference time.
- **Negative**: Adds dependencies on specific local feature extractors (ALIKED).

## Future Implications
This choice dictates the internal data flow of the `FineMatcher` abstraction. If an even faster or more robust matcher emerges, the modular design will allow us to wrap it in the `FineMatcher` interface and swap it without affecting the rest of the pipeline.
