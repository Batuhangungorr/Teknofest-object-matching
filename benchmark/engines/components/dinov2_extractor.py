"""benchmark.engines.components.dinov2_extractor — DINOv2 Feature Extractor.

Concrete implementation of the `FeatureExtractor` abstraction using
Facebook's DINOv2 models. It extracts both patch tokens and the CLS
token (global features) from an input image.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torchvision import transforms
from PIL import Image

from benchmark.engines.coarse_to_fine.feature_extractor import (
    FeatureExtractor,
    ImageFeatures,
)


class DINOv2FeatureExtractor(FeatureExtractor):
    """Concrete FeatureExtractor using DINOv2.
    
    Loads a pretrained DINOv2 model via PyTorch Hub and uses it to
    extract spatial patch features and a global CLS feature.
    Automatically handles 1-channel thermal images by duplicating
    channels to form a 3-channel RGB image.
    """

    def __init__(self) -> None:
        self._model: Optional[torch.nn.Module] = None
        self._device: torch.device = torch.device("cpu")
        self._transform: Optional[transforms.Compose] = None
        self._patch_size: int = 14
        self._model_name: str = "dinov2_vits14"
        self._input_size: int = 518

    @property
    def name(self) -> str:
        return f"DINOv2-ViT-{self._model_name.split('_')[-1]}"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        
        self._model_name = cfg.get("model_name", "dinov2_vits14")
        self._input_size = cfg.get("input_size", 518)
        
        # Determine device
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

        # Load model from torch hub
        try:
            self._model = torch.hub.load("facebookresearch/dinov2", self._model_name)
            self._model = self._model.to(self._device)
            self._model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load DINOv2 model '{self._model_name}': {e}") from e

        # Extract patch size from model name or configuration
        if "14" in self._model_name:
            self._patch_size = 14
        elif "8" in self._model_name:
            self._patch_size = 8
        else:
            self._patch_size = cfg.get("patch_size", 14)

        # Set up preprocessing
        # DINOv2 uses ImageNet defaults
        self._normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._to_tensor = transforms.ToTensor()

    def extract(self, image_path: Path) -> ImageFeatures:
        if self._model is None or self._normalize is None:
            raise RuntimeError("DINOv2FeatureExtractor must be initialized before calling extract().")
            
        resolved_path = self._validate_image_path(image_path)
        
        start_time = time.perf_counter()
        
        try:
            # 1. Load image and ensure RGB
            image = Image.open(resolved_path)
            image_size_hw = (image.height, image.width)
            
            # Convert to RGB (handles grayscale/Thermal implicitly)
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            # 2. Preprocess: Resize proportional to input_size (longest edge), then pad or just ensure multiples of 14
            # Actually, just resize the image so that both dimensions are multiples of 14, 
            # and the longest edge is approximately self._input_size
            w, h = image.width, image.height
            scale = self._input_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Make divisible by patch size
            new_w = (new_w // self._patch_size) * self._patch_size
            new_h = (new_h // self._patch_size) * self._patch_size
            
            # Ensure at least patch_size
            new_w = max(self._patch_size, new_w)
            new_h = max(self._patch_size, new_h)
            
            image_resized = image.resize((new_w, new_h), Image.BICUBIC)
            
            input_tensor = self._to_tensor(image_resized)
            input_tensor = self._normalize(input_tensor)
            # Add batch dimension and move to device
            input_tensor = input_tensor.unsqueeze(0).to(self._device)
            
            # 3. Forward pass
            with torch.no_grad():
                # DINOv2's forward_features returns a dict containing 'x_norm_clstoken' and 'x_norm_patchtokens'
                ret = self._model.forward_features(input_tensor)
                
            cls_token = ret['x_norm_clstoken'] # Shape: (1, embed_dim)
            patch_tokens = ret['x_norm_patchtokens'] # Shape: (1, num_patches, embed_dim)
            
            # Remove batch dimension
            cls_token = cls_token.squeeze(0).cpu()
            patch_tokens = patch_tokens.squeeze(0).cpu()
            
            # 4. Compute grid shape
            grid_h = new_h // self._patch_size
            grid_w = new_w // self._patch_size
            patch_grid_shape = (grid_h, grid_w)
            
            extraction_time_ms = (time.perf_counter() - start_time) * 1000.0
            
            # 5. Package results
            metadata = {
                "model_name": self._model_name,
                "input_size_hw": (new_h, new_w),
                "extraction_time_ms": extraction_time_ms,
                "device": str(self._device)
            }
            
            return ImageFeatures(
                patch_features=patch_tokens,
                patch_grid_shape=patch_grid_shape,
                source_image_path=resolved_path,
                image_size_hw=image_size_hw,
                global_features=cls_token,
                metadata=metadata
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract features from {resolved_path}: {e}") from e

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device.type == "mps":
            # MPS backend might not have empty_cache in all versions, handle gracefully
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
                
        self._transform = None
