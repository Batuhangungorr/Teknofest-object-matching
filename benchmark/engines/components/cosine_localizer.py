"""benchmark.engines.components.cosine_localizer — Cosine Heatmap Localizer.

Concrete implementation of the `CoarseLocalizer` abstraction that uses
cosine similarity between DINOv2 patch features to generate a heatmap
and extracts candidate regions via connected component analysis.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import logging

logger = logging.getLogger(__name__)

from benchmark.engines.coarse_to_fine.localizer import (
    CandidateRegion,
    CoarseLocalizer,
    LocalizationResult,
)
from benchmark.engines.coarse_to_fine.feature_extractor import ImageFeatures


class CosineHeatmapLocalizer(CoarseLocalizer):
    """Concrete CoarseLocalizer using Cosine Similarity.
    
    1. Computes global similarity using CLS tokens as a fast pre-filter.
    2. Computes patch-wise cosine similarity between reference and test features.
    3. Reshapes similarity into a spatial 2D heatmap.
    4. Thresholds the heatmap and extracts bounding boxes via connected components.
    5. Applies padding to candidate regions and returns them sorted by confidence.
    """

    def __init__(self) -> None:
        self._global_threshold: float = 0.3
        self._patch_threshold: float = 0.5
        self._padding_ratio: float = 0.2
        self._max_candidates: int = 1
        self._min_region_area_ratio: float = 0.001
        self._retain_heatmap: bool = False
        self._refine_with_grabcut: bool = True

    @property
    def name(self) -> str:
        return "CosineHeatmap"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._global_threshold = cfg.get("global_threshold", 0.3)
        self._patch_threshold = cfg.get("similarity_threshold", 0.5)
        self._padding_ratio = cfg.get("padding_ratio", 0.2)
        self._max_candidates = cfg.get("max_candidates", 1)
        self._min_region_area_ratio = cfg.get("min_region_area_ratio", 0.001)
        self._retain_heatmap = cfg.get("retain_heatmap", False)
        self._refine_with_grabcut = cfg.get("refine_with_grabcut", True)

    def localize(
        self,
        reference_features: ImageFeatures,
        test_features: ImageFeatures,
        reference_bounding_box: Optional[Any] = None,
    ) -> LocalizationResult:
        
        self._validate_features_compatible(reference_features, test_features)
        start_time = time.perf_counter()

        # Extract tensors (assuming PyTorch tensors from DINOv2Extractor)
        ref_global = reference_features.global_features
        test_global = test_features.global_features
        ref_patches = reference_features.patch_features
        test_patches = test_features.patch_features
        
        if reference_bounding_box is not None and ref_patches is not None:
            img_h, img_w = reference_features.image_size_hw
            grid_h, grid_w = reference_features.patch_grid_shape
            ratio_x = grid_w / img_w
            ratio_y = grid_h / img_h
            
            x_min = int(reference_bounding_box.xtl * ratio_x)
            y_min = int(reference_bounding_box.ytl * ratio_y)
            x_max = int(reference_bounding_box.xbr * ratio_x)
            y_max = int(reference_bounding_box.ybr * ratio_y)
            
            x_min = max(0, min(x_min, grid_w - 1))
            y_min = max(0, min(y_min, grid_h - 1))
            x_max = max(x_min + 1, min(x_max, grid_w))
            y_max = max(y_min + 1, min(y_max, grid_h))
            
            ref_patches_2d = ref_patches.view(grid_h, grid_w, -1)
            ref_patches = ref_patches_2d[y_min:y_max, x_min:x_max, :].reshape(-1, ref_patches.shape[-1])

        # Ensure tensors are on the same device
        device = test_patches.device if hasattr(test_patches, 'device') else torch.device("cpu")
        if hasattr(ref_patches, 'to'):
            ref_patches = ref_patches.to(device)
            if ref_global is not None:
                ref_global = ref_global.to(device)
        

        global_similarity = None

        # 1. Global Similarity Pre-filter (Phase 3)
        if ref_global is not None and test_global is not None:
            # L2 normalize
            ref_global_norm = F.normalize(ref_global.float(), p=2, dim=-1)
            test_global_norm = F.normalize(test_global.float(), p=2, dim=-1)
            # Compute cosine similarity
            global_sim = torch.dot(ref_global_norm.squeeze(), test_global_norm.squeeze()).item()
            global_similarity = float(global_sim)

            if global_similarity < self._global_threshold:
                # Short-circuit if global similarity is too low
                return self._build_empty_result(
                    test_features,
                    start_time,
                    global_similarity,
                    reason="global_similarity_below_threshold"
                )

        # 2. Patch Similarity Computation (Phase 3 & 4)
        # L2 normalize patches
        ref_patches_norm = F.normalize(ref_patches.float(), p=2, dim=-1) # (num_ref_patches, embed_dim)
        test_patches_norm = F.normalize(test_patches.float(), p=2, dim=-1) # (num_test_patches, embed_dim)
        
        # Max similarity across all reference patches to be robust
        sim_matrix = torch.matmul(test_patches_norm, ref_patches_norm.transpose(0, 1))
        patch_sims, _ = sim_matrix.max(dim=-1)

        # 3. Reshape to spatial heatmap
        grid_h, grid_w = test_features.patch_grid_shape
        heatmap_tensor = patch_sims.view(grid_h, grid_w)
        heatmap_np = heatmap_tensor.cpu().numpy()

        # Extract Candidate Regions
        max_val = float(heatmap_np.max())
        
        candidates = []
        img_h, img_w = test_features.image_size_hw
        
        # Mapping ratio from heatmap grid to original image
        ratio_y = img_h / grid_h
        ratio_x = img_w / grid_w
        
        # If we have a reference bounding box, we know the object's approximate size!
        # Find the peak of the heatmap and place a box of that size around it.
        if reference_bounding_box is not None:
            ref_w = reference_bounding_box.xbr - reference_bounding_box.xtl
            ref_h = reference_bounding_box.ybr - reference_bounding_box.ytl
            
            # Find peak location in heatmap
            peak_y, peak_x = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
            
            # Sub-patch interpolation (center of mass in 3x3 neighborhood)
            h_grid, w_grid = heatmap_np.shape
            y_min_grid = max(0, peak_y - 1)
            y_max_grid = min(h_grid - 1, peak_y + 1)
            x_min_grid = max(0, peak_x - 1)
            x_max_grid = min(w_grid - 1, peak_x + 1)
            
            neighborhood = heatmap_np[y_min_grid:y_max_grid+1, x_min_grid:x_max_grid+1]
            
            # Ensure non-negative weights for center of mass
            weights = np.maximum(0, neighborhood)
            sum_weights = np.sum(weights)
            
            if sum_weights > 0:
                y_indices, x_indices = np.indices(weights.shape)
                # Offset indices by the min bounds to get global grid coordinates
                y_indices = y_indices + y_min_grid
                x_indices = x_indices + x_min_grid
                
                interp_y = np.sum(y_indices * weights) / sum_weights
                interp_x = np.sum(x_indices * weights) / sum_weights
            else:
                interp_y, interp_x = float(peak_y), float(peak_x)
            
            # Center in image coordinates
            center_x = (interp_x + 0.5) * ratio_x
            center_y = (interp_y + 0.5) * ratio_y
            
            x_min = max(0.0, center_x - ref_w / 2.0)
            y_min = max(0.0, center_y - ref_h / 2.0)
            x_max = min(float(img_w), center_x + ref_w / 2.0)
            y_max = min(float(img_h), center_y + ref_h / 2.0)
            
            candidates.append(CandidateRegion(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                confidence=max_val,
                metadata={"peak_similarity": max_val}
            ))
        else:
            # Fallback to dynamic thresholding if no reference box
            dynamic_threshold = max_val * 0.98
            binary_mask = (heatmap_np >= dynamic_threshold).astype(np.uint8)
            
            if binary_mask.sum() == 0:
                return self._build_empty_result(
                    test_features, start_time, global_similarity, heatmap_np, "no_patches_above_threshold"
                )
                
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            for i in range(1, num_labels):
                area_grid = stats[i, cv2.CC_STAT_AREA]
                x_grid = stats[i, cv2.CC_STAT_LEFT]
                y_grid = stats[i, cv2.CC_STAT_TOP]
                w_grid = stats[i, cv2.CC_STAT_WIDTH]
                h_grid = stats[i, cv2.CC_STAT_HEIGHT]
                
                x_min = x_grid * ratio_x
                y_min = y_grid * ratio_y
                x_max = (x_grid + w_grid) * ratio_x
                y_max = (y_grid + h_grid) * ratio_y
                
                conf = float(heatmap_np[labels == i].mean())
                candidates.append(CandidateRegion(
                    x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, confidence=conf
                ))

        # Sort candidates by confidence descending
        candidates.sort(key=lambda c: c.confidence if c.confidence else 0.0, reverse=True)
        candidates = candidates[:self._max_candidates]
        
        # Assign ranks
        for idx, c in enumerate(candidates):
            c.metadata["rank"] = idx
            
        extraction_time_ms = (time.perf_counter() - start_time) * 1000.0

        return LocalizationResult(
            candidates=candidates,
            test_image_path=test_features.source_image_path,
            test_image_size_hw=test_features.image_size_hw,
            global_similarity=global_similarity,
            heatmap=heatmap_np if self._retain_heatmap else None,
            heatmap_shape=(grid_h, grid_w),
            metadata={
                "localization_time_ms": extraction_time_ms,
                "num_patches_compared": int(ref_patches.shape[0] * test_patches.shape[0]),
                "threshold_used": self._patch_threshold,
                "algorithm": self.name
            }
        )
        
    def _build_empty_result(
        self,
        test_features: ImageFeatures,
        start_time: float,
        global_sim: Optional[float] = None,
        heatmap_np: Optional[np.ndarray] = None,
        reason: str = "unknown"
    ) -> LocalizationResult:
        grid_h, grid_w = test_features.patch_grid_shape
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        return LocalizationResult(
            candidates=[],
            test_image_path=test_features.source_image_path,
            test_image_size_hw=test_features.image_size_hw,
            global_similarity=global_sim,
            heatmap=heatmap_np if self._retain_heatmap else None,
            heatmap_shape=(grid_h, grid_w),
            metadata={
                "localization_time_ms": elapsed_ms,
                "reason": reason,
                "algorithm": self.name
            }
        )

    def cleanup(self) -> None:
        pass
