# Literature

This document summarizes current state-of-the-art literature relevant to one-shot object matching and explains the architectural choices for this project.

## Models Overview

- **DINOv2**: A self-supervised Vision Transformer (ViT) by Meta. **Strengths**: Exceptional at producing robust, dense semantic features without task-specific fine-tuning. **Weaknesses**: Computationally heavy for dense local feature extraction at high resolutions.
- **ALIKED**: A lightweight local feature detector and descriptor. **Strengths**: Fast and provides highly repeatable keypoints. **Weaknesses**: Requires a separate matcher.
- **LightGlue**: A deep matcher that replaces SuperGlue. **Strengths**: Highly efficient, accurate, and adaptable to various local features (like ALIKED). Adapts computational complexity based on difficulty.
- **SuperPoint & SuperGlue**: Older standards for feature detection and matching. LightGlue is effectively a faster, more accurate successor to SuperGlue.
- **LoFTR / EfficientLoFTR**: Detector-free local feature matchers. **Strengths**: Excellent in low-texture environments. **Weaknesses**: Computationally expensive to run across full high-resolution test frames without prior localization.
- **GeM & CosPlace**: Techniques primarily for visual place recognition. Excellent for global image descriptors but less suited for precise local object bounding box projection.
- **RANSAC**: The standard geometric verification algorithm.

## Architectural Selections

### Why DINOv2?
Selected for the **Coarse Localizer** phase. DINOv2 excels at zero-shot semantic similarity. By using it on downscaled or patch-level data, we can accurately and robustly identify the general area of the target object, even under severe appearance changes.

### Why ALIKED + LightGlue?
Selected for the **Fine Matcher** phase. Once the coarse region is identified, we need pixel-perfect precision. ALIKED provides robust keypoints, and LightGlue offers state-of-the-art matching efficiency. Using them together inside a cropped candidate region balances speed and accuracy far better than running LoFTR on a full 4K drone frame.

### Why a Unified RGB + Thermal Pipeline?
Research shows that models like DINOv2, trained extensively on diverse RGB data, extract structural and semantic features that generalize surprisingly well to thermal imagery when properly preprocessed. Maintaining a single pipeline guarantees that architectural optimizations benefit both domains simultaneously, reducing maintenance overhead and providing a clean ablation baseline for future modality-specific research.
