"""benchmark.engines.components.aliked_matcher — ALIKED + LightGlue matcher.

Concrete implementation of the `FineMatcher` abstraction that uses
Kornia's ALIKED detector and LightGlue matcher.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torchvision import transforms
from PIL import Image

from benchmark.engines.coarse_to_fine.matcher import (
    FineMatcher,
    Keypoint,
    KeypointSet,
    MatchingResult,
)
from benchmark.data_models import BoundingBox
from benchmark.engines.coarse_to_fine.feature_extractor import ImageFeatures
from benchmark.engines.coarse_to_fine.localizer import CandidateRegion

try:
    from kornia.feature import ALIKED, LightGlue
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False


class AlikedLightGlueMatcher(FineMatcher):
    """Concrete FineMatcher using ALIKED and LightGlue.
    
    1. Detects keypoints and extracts descriptors using ALIKED.
    2. Matches descriptors using LightGlue.
    3. Handles reference caching for fast repeated lookups.
    """

    def __init__(self) -> None:
        self._device: torch.device = torch.device("cpu")
        self._max_keypoints: int = 2048
        self._aliked_model_name: str = "aliked-n16"
        self._detector: Optional[torch.nn.Module] = None
        self._matcher: Optional[torch.nn.Module] = None
        self._reference_keypoints: Optional[KeypointSet] = None
        self._reference_kps_tensor_crop: Optional[torch.Tensor] = None
        self._reference_crop_size_hw: Optional[Tuple[int, int]] = None

    @property
    def name(self) -> str:
        return "ALIKED+LightGlue"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not KORNIA_AVAILABLE:
            raise RuntimeError("Kornia is not installed. Please `pip install kornia` to use AlikedLightGlueMatcher.")
            
        cfg = config or {}
        
        device_str = cfg.get("device")
        if device_str:
            self._device = torch.device(device_str)
        else:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
                
        self._max_keypoints = cfg.get("max_keypoints", 2048)
        self._aliked_model_name = cfg.get("detector_name", "aliked-n16")

        try:
            # Initialize ALIKED
            self._detector = ALIKED(model_name=self._aliked_model_name, max_num_keypoints=self._max_keypoints).to(self._device)
            self._detector.eval()
            
            # Initialize LightGlue with ALIKED configuration
            self._matcher = LightGlue("aliked").to(self._device)
            self._matcher.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load ALIKED/LightGlue models: {e}") from e

    def _load_and_preprocess_image(self, path: Path, crop_box: Optional[Tuple[float, float, float, float]] = None) -> torch.Tensor:
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        if crop_box is not None:
            # crop_box is (x_min, y_min, x_max, y_max)
            image = image.crop(crop_box)
            
        transform = transforms.ToTensor()
        tensor = transform(image).unsqueeze(0).to(self._device) # (1, 3, H, W)
        return tensor

    def set_reference(
        self,
        reference_features: ImageFeatures,
        reference_bounding_box: Optional[BoundingBox] = None,
    ) -> None:
        self._validate_reference_set(reference_features)
        
        if self._detector is None:
            raise RuntimeError("Matcher not initialized. Call initialize() first.")
            
        start_time = time.perf_counter()
        
        path = reference_features.source_image_path
        if reference_bounding_box is not None:
            crop_box = (
                reference_bounding_box.xtl,
                reference_bounding_box.ytl,
                reference_bounding_box.xbr,
                reference_bounding_box.ybr,
            )
            x_offset, y_offset = reference_bounding_box.xtl, reference_bounding_box.ytl
        else:
            crop_box = None
            x_offset, y_offset = 0.0, 0.0
            
        img_tensor = self._load_and_preprocess_image(path, crop_box)
        
        with torch.no_grad():
            features = self._detector(img_tensor)
            
        kps_tensor = features[0].keypoints  # (N, 2)
        desc_tensor = features[0].descriptors # (N, D)
        
        self._reference_kps_tensor_crop = kps_tensor
        self._reference_crop_size_hw = (img_tensor.shape[2], img_tensor.shape[3]) # (H, W)
        
        keypoints = []
        for i in range(kps_tensor.shape[0]):
            x, y = kps_tensor[i].tolist()
            keypoints.append(Keypoint(
                x=float(x + x_offset),
                y=float(y + y_offset),
                descriptor=None, 
            ))
        print(f"ALIKED extracted {len(keypoints)} reference keypoints from crop size {img_tensor.shape[2]}x{img_tensor.shape[3]}")
            
        extraction_time_ms = (time.perf_counter() - start_time) * 1000.0
            
        self._reference_keypoints = KeypointSet(
            keypoints=keypoints,
            image_path=path,
            image_size_hw=reference_features.image_size_hw,
            descriptors=desc_tensor,
            metadata={
                "detector_name": self._aliked_model_name,
                "detection_time_ms": extraction_time_ms
            }
        )

    def match(
        self,
        test_features: ImageFeatures,
        candidate_region: CandidateRegion,
    ) -> MatchingResult:
        
        if self._reference_keypoints is None:
            raise RuntimeError("Reference not set. Call set_reference() first.")
            
        if self._matcher is None or self._detector is None:
            raise RuntimeError("Matcher not initialized. Call initialize() first.")
            
        start_time = time.perf_counter()
        
        path = test_features.source_image_path
        crop_box = (candidate_region.x_min, candidate_region.y_min, candidate_region.x_max, candidate_region.y_max)
        
        img_tensor = self._load_and_preprocess_image(path, crop_box)
        
        with torch.no_grad():
            features = self._detector(img_tensor)
            
        kps_tensor = features[0].keypoints
        desc_tensor = features[0].descriptors
        
        keypoints = []
        x_offset, y_offset = candidate_region.x_min, candidate_region.y_min
        for i in range(kps_tensor.shape[0]):
            x_local, y_local = kps_tensor[i].tolist()
            keypoints.append(Keypoint(
                x=float(x_local + x_offset),
                y=float(y_local + y_offset),
                descriptor=None,
            ))
            
        test_keypoint_set = KeypointSet(
            keypoints=keypoints,
            image_path=path,
            image_size_hw=test_features.image_size_hw,
            descriptors=desc_tensor,
        )
        
        if self._reference_keypoints.num_keypoints > 0 and test_keypoint_set.num_keypoints > 0:
            ref_h, ref_w = self._reference_crop_size_hw
            test_h, test_w = img_tensor.shape[2], img_tensor.shape[3]
            
            features1 = {
                'keypoints': self._reference_kps_tensor_crop.unsqueeze(0),
                'descriptors': self._reference_keypoints.descriptors.unsqueeze(0),
                'image_size': torch.tensor([[ref_w, ref_h]], device=self._device)
            }
            
            features2 = {
                'keypoints': kps_tensor.unsqueeze(0),
                'descriptors': desc_tensor.unsqueeze(0),
                'image_size': torch.tensor([[test_w, test_h]], device=self._device)
            }
            
            print(f"ALIKED test keypoints: {kps_tensor.shape[0]}, Ref keypoints: {self._reference_kps_tensor_crop.shape[0]}")
            
            with torch.no_grad():
                out = self._matcher({'image0': features1, 'image1': features2})
                
            m = out["matches"][0] # (M, 2)
            print(f"LightGlue found {m.shape[0]} matches")
            
            ref_indices = m[:, 0].tolist()
            test_indices = m[:, 1].tolist()
            confidences = out["scores"][0].tolist()
        else:
            ref_indices = []
            test_indices = []
            confidences = []
            
        matching_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        return MatchingResult(
            reference_keypoints=self._reference_keypoints,
            test_keypoints=test_keypoint_set,
            reference_indices=ref_indices,
            test_indices=test_indices,
            candidate_region=candidate_region,
            match_confidences=confidences,
            overall_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            metadata={
                "matcher_name": self.name,
                "matching_time_ms": matching_time_ms,
                "crop_size_hw": (img_tensor.shape[2], img_tensor.shape[3])
            }
        )

    def cleanup(self) -> None:
        if self._detector is not None:
            del self._detector
            self._detector = None
            
        if self._matcher is not None:
            del self._matcher
            self._matcher = None
            
        self._reference_keypoints = None
        self._reference_kps_tensor_crop = None
        self._reference_crop_size_hw = None
        
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device.type == "mps":
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
