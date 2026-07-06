"""benchmark.evaluate — Unified evaluation script for RGB and Thermal datasets."""

import argparse
import logging
import sys
from pathlib import Path

from benchmark.data_models import ValidationDataset
from benchmark.loader import load_validation_dataset
from benchmark.engine import MatchingEngine
from benchmark.engines.coarse_to_fine.engine import CoarseToFineEngine
from benchmark.engines.components.dinov2_extractor import DINOv2FeatureExtractor
from benchmark.engines.components.cosine_localizer import CosineHeatmapLocalizer
from benchmark.engines.components.aliked_matcher import AlikedLightGlueMatcher
from benchmark.engines.components.ransac_verifier import RansacHomographyVerifier
from benchmark.runner import BenchmarkRunner
from benchmark.metrics import MetricsCalculator

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

def build_engine() -> MatchingEngine:
    return CoarseToFineEngine(
        feature_extractor=DINOv2FeatureExtractor(),
        coarse_localizer=CosineHeatmapLocalizer(),
        fine_matcher=AlikedLightGlueMatcher(),
        geometric_verifier=RansacHomographyVerifier(),
    )

def main():
    parser = argparse.ArgumentParser(description="Evaluate CoarseToFineEngine on Validation Datasets.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the validation dataset root (e.g., Validation/RGB)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda or cpu)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for True Positive")
    parser.add_argument("--sequential-tracking", action="store_true", help="Enable sequential tracking by updating the template dynamically")
    
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger("benchmark.evaluate")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = load_validation_dataset(dataset_path)
    
    logger.info(f"Building engine...")
    engine = build_engine()
    
    config = {
        "feature_extractor": {"device": args.device, "input_size": 1008},
        "coarse_localizer": {
            "global_threshold": 0.3, 
            "similarity_threshold": 0.45,
            "padding_ratio": 0.0,
            "refine_with_grabcut": False
        },
        "fine_matcher": {"device": args.device, "max_keypoints": 1000},
    }
    
    runner = BenchmarkRunner(engine=engine, dataset=dataset, sequential_tracking=args.sequential_tracking)
    
    logger.info("Starting BenchmarkRunner...")
    predictions = runner.run(config)
    
    logger.info("Computing metrics...")
    calculator = MetricsCalculator(iou_threshold=args.iou_threshold)
    report = calculator.evaluate(dataset, predictions)
    
    print("\n" + "="*50)
    print(" " * 15 + "EVALUATION REPORT")
    print("="*50)
    print(f"Engine: {predictions.engine_name}")
    print(f"Dataset: {dataset.root_dir.name}")
    print(f"Total Sets: {dataset.num_sets}")
    print(f"Total Images: {dataset.total_images}")
    print("-" * 50)
    
    per_set_stats = report.get("per_set", {})
    for set_name, stats in per_set_stats.items():
        print(f"\nSET: {set_name}")
        print(f"  Precision: {stats['Precision']:.4f}")
        print(f"  Recall:    {stats['Recall']:.4f}")
        print(f"  F1 Score:  {stats['F1 Score']:.4f}")
        print(f"  Accuracy:  {stats['Accuracy']:.4f}")
        print(f"  Mean IoU:  {stats['Mean IoU (Successes)']:.4f}")
        print("  Counts:")
        for k, v in stats['Counts'].items():
            print(f"    {k}: {v}")
            
    print("-" * 50)
    
    global_stats = report["global"]
    print("\nGLOBAL METRICS:")
    print(f"  Precision: {global_stats['Precision']:.4f}")
    print(f"  Recall:    {global_stats['Recall']:.4f}")
    print(f"  F1 Score:  {global_stats['F1 Score']:.4f}")
    print(f"  Accuracy:  {global_stats['Accuracy']:.4f}")
    print(f"  Mean IoU:  {global_stats['Mean IoU (Successes)']:.4f}")
    print("\n  Counts:")
    for k, v in global_stats['Counts'].items():
        print(f"    {k}: {v}")
        
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
