"""Test script for Cosine Heatmap Localizer."""

import logging
from pathlib import Path

import torch

from benchmark.engines.components.cosine_localizer import CosineHeatmapLocalizer
from benchmark.engines.coarse_to_fine.feature_extractor import ImageFeatures

logging.basicConfig(level=logging.INFO)

def main():
    localizer = CosineHeatmapLocalizer()
    print(f"Localizer Name: {localizer.name}")
    
    print("Initializing...")
    localizer.initialize({
        "global_threshold": 0.3,
        "similarity_threshold": 0.5,
        "padding_ratio": 0.2,
        "max_candidates": 2,
        "retain_heatmap": True
    })
    
    # Create dummy DINOv2 features for reference and test
    embed_dim = 384
    ref_global = torch.rand(1, embed_dim)
    ref_patches = torch.rand(196, embed_dim) # 14x14 patches
    
    # Let's make a test image with high similarity at a specific location
    test_global = ref_global.clone() # High global similarity
    test_patches = torch.rand(3600, embed_dim) # 60x60 patches
    
    # Plant a highly similar patch at row 30, col 30 -> flat index 30*60 + 30 = 1830
    test_patches[1830] = ref_patches[0]
    
    ref_features = ImageFeatures(
        patch_features=ref_patches,
        patch_grid_shape=(14, 14),
        source_image_path=Path("dummy_ref.jpg"),
        image_size_hw=(196, 196),
        global_features=ref_global
    )
    
    test_features = ImageFeatures(
        patch_features=test_patches,
        patch_grid_shape=(60, 60),
        source_image_path=Path("dummy_test.jpg"),
        image_size_hw=(840, 840),
        global_features=test_global
    )
    
    print("Localizing...")
    try:
        result = localizer.localize(ref_features, test_features)
        print("Localization successful!")
        print(f"Result: {result}")
        print(f"Global similarity: {result.global_similarity}")
        for idx, candidate in enumerate(result.candidates):
            print(f"Candidate {idx+1}: {candidate}")
    finally:
        localizer.cleanup()
        print("Cleanup completed.")

if __name__ == "__main__":
    main()
