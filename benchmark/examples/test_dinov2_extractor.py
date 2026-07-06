"""Test script for DINOv2 Feature Extractor."""

import logging
from pathlib import Path
from PIL import Image

from benchmark.engines.components.dinov2_extractor import DINOv2FeatureExtractor

logging.basicConfig(level=logging.INFO)

def main():
    extractor = DINOv2FeatureExtractor()
    print(f"Extractor Name: {extractor.name}")
    
    print("Initializing...")
    extractor.initialize({"model_name": "dinov2_vits14", "input_size": 518})
    
    # Create a dummy image for testing
    dummy_img_path = Path("dummy_test_image.jpg")
    img = Image.new('RGB', (640, 480), color = 'red')
    img.save(dummy_img_path)
    
    print(f"Extracting features from {dummy_img_path}...")
    try:
        features = extractor.extract(dummy_img_path)
        print("Extraction successful!")
        print(f"Image Features Representation: {features}")
        print(f"Global features shape: {features.global_features.shape}")
        print(f"Patch features shape: {features.patch_features.shape}")
        print(f"Patch grid shape: {features.patch_grid_shape}")
    finally:
        if dummy_img_path.exists():
            dummy_img_path.unlink()
        extractor.cleanup()
        print("Cleanup completed.")

if __name__ == "__main__":
    main()
