"""benchmark.examples.test_pipeline — End-to-end integration test."""

import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from benchmark.data_models import BoundingBox
from benchmark.engines.coarse_to_fine.engine import CoarseToFineEngine
from benchmark.engines.components.dinov2_extractor import DINOv2FeatureExtractor
from benchmark.engines.components.cosine_localizer import CosineHeatmapLocalizer
from benchmark.engines.components.aliked_matcher import AlikedLightGlueMatcher
from benchmark.engines.components.ransac_verifier import RansacHomographyVerifier

def create_synthetic_images(output_dir: Path) -> tuple[Path, Path, BoundingBox]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_path = output_dir / "reference.jpg"
    test_path = output_dir / "test.jpg"

    # Create a reference image (a colorful shape on a solid background)
    ref_img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(ref_img, (150, 150), 50, (255, 100, 50), -1)
    cv2.rectangle(ref_img, (100, 100), (200, 200), (50, 255, 100), 5)
    cv2.putText(ref_img, "TEST", (110, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save reference
    Image.fromarray(ref_img).save(ref_path)

    # Create a test image (larger, different background)
    test_img = np.zeros((800, 800, 3), dtype=np.uint8)
    
    # Insert the reference image into the test image at (200, 300)
    x_offset, y_offset = 200, 300
    test_img[y_offset:y_offset+300, x_offset:x_offset+300] = ref_img
    
    # Save test
    Image.fromarray(test_img).save(test_path)
    
    # Ground truth box in reference image (the whole image for this simple test)
    ref_box = BoundingBox(label="object", xtl=0.0, ytl=0.0, xbr=300.0, ybr=300.0)
    
    return ref_path, test_path, ref_box

def main():
    print("Setting up end-to-end pipeline...")
    engine = CoarseToFineEngine(
        feature_extractor=DINOv2FeatureExtractor(),
        coarse_localizer=CosineHeatmapLocalizer(),
        fine_matcher=AlikedLightGlueMatcher(),
        geometric_verifier=RansacHomographyVerifier(),
    )
    
    engine.initialize({
        "feature_extractor": {"device": "cpu"},
        "fine_matcher": {"device": "cpu", "max_keypoints": 500},
    })
    
    print("Generating synthetic images...")
    output_dir = Path("test_output")
    ref_path, test_path, ref_box = create_synthetic_images(output_dir)
    
    print("Setting reference...")
    start_time = time.perf_counter()
    engine.set_reference(ref_path, ref_box)
    print(f"Reference set in {time.perf_counter() - start_time:.2f}s")
    
    print("Running detection on test image...")
    start_time = time.perf_counter()
    prediction = engine.detect(test_path)
    print(f"Detection completed in {time.perf_counter() - start_time:.2f}s")
    
    print("\n--- Detection Result ---")
    print(f"Result: {prediction.result.name}")
    print(f"Confidence: {prediction.confidence}")
    if prediction.predicted_box:
        print(f"Predicted BBox: {prediction.predicted_box}")
    print(f"Metadata: {prediction.metadata}")
    
    engine.cleanup()
    print("Pipeline cleanup complete.")

if __name__ == "__main__":
    main()
