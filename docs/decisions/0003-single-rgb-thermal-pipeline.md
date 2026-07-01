# ADR 0003: Single RGB + Thermal Pipeline

**Status:** Accepted
**Date:** 2026-06-27

## Context
The dataset contains both RGB (visible spectrum) and Thermal imagery. The standard approach in many competitions is to train separate models or develop separate algorithmic pipelines for each modality.

## Decision
We decided to use a single, unified pipeline for BOTH RGB and Thermal datasets. Thermal images will be preprocessed (e.g., duplicated across 3 channels, normalized) to mimic the input structure expected by RGB-trained foundation models. The matching architecture will not contain `if thermal:` branches.

## Alternatives
- **Two separate pipelines**: One specifically optimized for thermal, one for RGB.
- **Modality-specific weights**: Training separate LightGlue/ALIKED weights for the thermal domain.

## Consequences
- **Positive**: Halves the maintenance burden. Capitalizes on the zero-shot structural generalization of large foundation models (like DINOv2).
- **Negative**: Might theoretically sacrifice some maximum possible accuracy on thermal images compared to a purely thermal-trained model.

## Future Implications
Any optimization made to the single pipeline automatically benefits both modalities. If benchmark results later prove an insurmountable gap, this decision can be revisited, but it establishes a strong, unified baseline.
