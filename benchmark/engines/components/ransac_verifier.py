"""benchmark.engines.components.ransac_verifier — Geometric verification.

Concrete implementation of the `GeometricVerifier` abstraction using OpenCV's RANSAC
homography estimation.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from benchmark.data_models import BoundingBox
from benchmark.engines.coarse_to_fine.localizer import CandidateRegion
from benchmark.engines.coarse_to_fine.matcher import MatchingResult
from benchmark.engines.coarse_to_fine.verifier import (
    GeometricVerifier,
    VerificationResult,
)


class RansacHomographyVerifier(GeometricVerifier):
    """Concrete GeometricVerifier using OpenCV RANSAC.
    
    1. Extracts paired point coordinates.
    2. Runs cv2.findHomography with RANSAC.
    3. Identifies inliers and calculates reprojection error.
    4. Projects the reference bounding box to the test image.
    """

    def __init__(self) -> None:
        self._reproj_threshold: float = 5.0
        self._min_inliers: int = 4
        self._ransac_confidence: float = 0.99
        self._ransac_max_iters: int = 2000

    @property
    def name(self) -> str:
        return "RANSAC-Homography"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._reproj_threshold = cfg.get("ransac_reproj_threshold", 5.0)
        self._min_inliers = max(4, cfg.get("min_inliers", 4)) # Need at least 4 for homography
        self._ransac_confidence = cfg.get("ransac_confidence", 0.99)
        self._ransac_max_iters = cfg.get("ransac_max_iters", 2000)

    def verify(
        self,
        matching_result: MatchingResult,
        reference_bounding_box: BoundingBox,
    ) -> VerificationResult:
        
        self._validate_has_matches(matching_result)
        start_time = time.perf_counter()
        
        candidate = matching_result.candidate_region
        test_img_path = matching_result.test_keypoints.image_path
        test_img_size = matching_result.test_keypoints.image_size_hw
        
        # 1. Gather correspondences
        if not self._validate_minimum_correspondences(matching_result, 4):
            return self._build_failure_result(
                candidate, test_img_path, test_img_size, start_time,
                "insufficient_correspondences"
            )
            
        ref_pts = np.array(matching_result.matched_ref_coordinates, dtype=np.float32)
        test_pts = np.array(matching_result.matched_test_coordinates, dtype=np.float32)

        # 3. Robust model fitting (RANSAC)
        H, inlier_mask = cv2.findHomography(
            ref_pts,
            test_pts,
            cv2.RANSAC,
            self._reproj_threshold,
            confidence=self._ransac_confidence,
            maxIters=self._ransac_max_iters
        )
        
        if H is None or inlier_mask is None:
            return self._build_failure_result(
                candidate, test_img_path, test_img_size, start_time,
                "ransac_failed_to_converge"
            )
            
        inlier_mask = inlier_mask.ravel().astype(bool)
        num_inliers = int(inlier_mask.sum())
        num_outliers = len(inlier_mask) - num_inliers
        
        # 4. Apply acceptance thresholds
        if num_inliers < self._min_inliers:
            return self._build_failure_result(
                candidate, test_img_path, test_img_size, start_time,
                "insufficient_inliers",
                inlier_mask=inlier_mask, num_inliers=num_inliers, num_outliers=num_outliers
            )
            
        # 5. Project the bounding box
        corners = np.array(self._reference_bbox_corners(reference_bounding_box), dtype=np.float32).reshape(1, -1, 2)
        projected_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        
        # Compute axis-aligned bounding box
        x_coords = projected_corners[:, 0]
        y_coords = projected_corners[:, 1]
        
        raw_bbox = BoundingBox(
            label=reference_bounding_box.label,
            xtl=float(x_coords.min()),
            ytl=float(y_coords.min()),
            xbr=float(x_coords.max()),
            ybr=float(y_coords.max()),
        )
        
        verified_bbox = self._clip_bounding_box_to_image(raw_bbox, test_img_size)
        
        # Calculate mean reprojection error
        ref_inliers = ref_pts[inlier_mask].reshape(-1, 1, 2)
        test_inliers = test_pts[inlier_mask].reshape(-1, 2)
        proj_ref_inliers = cv2.perspectiveTransform(ref_inliers, H).reshape(-1, 2)
        errors = np.linalg.norm(proj_ref_inliers - test_inliers, axis=1)
        mean_reproj_err = float(errors.mean())
        
        # 6. Score confidence
        inlier_ratio = num_inliers / len(ref_pts)
        confidence = inlier_ratio * min(1.0, num_inliers / 20.0) 
        
        verification_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        # 7. Assemble result
        return VerificationResult(
            success=True,
            candidate_region=candidate,
            test_image_path=test_img_path,
            test_image_size_hw=test_img_size,
            verified_bounding_box=verified_bbox,
            homography=H,
            inlier_mask=inlier_mask,
            num_inliers=num_inliers,
            num_outliers=num_outliers,
            confidence=confidence,
            mean_reprojection_error=mean_reproj_err,
            metadata={
                "verifier_name": self.name,
                "verification_time_ms": verification_time_ms,
                "model_type": "homography"
            }
        )

    def _build_failure_result(
        self,
        candidate: CandidateRegion,
        test_img_path: Path,
        test_img_size: Tuple[int, int],
        start_time: float,
        reason: str,
        inlier_mask: Optional[np.ndarray] = None,
        num_inliers: int = 0,
        num_outliers: int = 0,
    ) -> VerificationResult:
        verification_time_ms = (time.perf_counter() - start_time) * 1000.0
        return VerificationResult(
            success=False,
            candidate_region=candidate,
            test_image_path=test_img_path,
            test_image_size_hw=test_img_size,
            inlier_mask=inlier_mask,
            num_inliers=num_inliers,
            num_outliers=num_outliers,
            metadata={
                "verifier_name": self.name,
                "verification_time_ms": verification_time_ms,
                "failure_reason": reason
            }
        )

    def cleanup(self) -> None:
        pass
