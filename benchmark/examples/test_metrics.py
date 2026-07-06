"""benchmark.examples.test_metrics — Test metrics logic."""

from pathlib import Path

from benchmark.data_models import BoundingBox, ImageAnnotation, ReferenceSet, ValidationDataset
from benchmark.prediction_models import BenchmarkPredictions, DetectionPrediction, MatchResult, SetPredictions
from benchmark.metrics import MetricsCalculator

def test_metrics():
    # Setup Ground Truth
    img1 = ImageAnnotation(filename="img1.jpg", filepath=Path("img1.jpg"), width=100, height=100, boxes=[
        BoundingBox(label="obj", xtl=10, ytl=10, xbr=50, ybr=50) # 40x40 = 1600 area
    ])
    img2 = ImageAnnotation(filename="img2.jpg", filepath=Path("img2.jpg"), width=100, height=100, boxes=[
        BoundingBox(label="obj", xtl=10, ytl=10, xbr=50, ybr=50)
    ])
    img3_empty = ImageAnnotation(filename="img3.jpg", filepath=Path("img3.jpg"), width=100, height=100, boxes=[])
    img4_error = ImageAnnotation(filename="img4.jpg", filepath=Path("img4.jpg"), width=100, height=100, boxes=[
        BoundingBox(label="obj", xtl=10, ytl=10, xbr=50, ybr=50)
    ])

    ref_set = ReferenceSet(
        name="ref01",
        root_dir=Path("ref01"),
        reference_image_path=Path("ref01/ref.jpg"),
        images_dir=Path("ref01/images"),
        annotations=[img1, img2, img3_empty, img4_error]
    )
    dataset = ValidationDataset(root_dir=Path("dataset"), reference_sets=[ref_set])

    # Setup Predictions
    # img1: TP (>0.75) because perfectly matches (IoU=1.0)
    p1 = DetectionPrediction(
        image_filename="img1.jpg", reference_set="ref01", result=MatchResult.SUCCESS,
        predicted_box=BoundingBox(label="obj", xtl=10, ytl=10, xbr=50, ybr=50)
    )
    
    # img2: TP (normal, IoU ~0.53) - shifted by 10 pixels
    p2 = DetectionPrediction(
        image_filename="img2.jpg", reference_set="ref01", result=MatchResult.SUCCESS,
        predicted_box=BoundingBox(label="obj", xtl=20, ytl=20, xbr=60, ybr=60)
    )
    
    # img3_empty: TN
    p3 = DetectionPrediction(image_filename="img3.jpg", reference_set="ref01", result=MatchResult.NO_MATCH)
    
    # img4_error: FN (Error)
    p4 = DetectionPrediction(image_filename="img4.jpg", reference_set="ref01", result=MatchResult.ERROR)

    set_preds = SetPredictions(set_name="ref01", engine_name="TestEngine", predictions=[p1, p2, p3, p4])
    preds = BenchmarkPredictions(engine_name="TestEngine", set_predictions=[set_preds])

    # Evaluate
    calc = MetricsCalculator()
    report = calc.evaluate(dataset, preds)

    import pprint
    pprint.pprint(report)
    
    # Asserts
    global_stats = report["global"]
    counts = global_stats["Counts"]
    assert counts["TP (>0.75)"] == 1
    assert counts["TP"] == 0
    assert counts["FP"] == 1
    assert counts["FN (Error)"] == 1
    
    print("\nAll assertions passed!")

if __name__ == "__main__":
    test_metrics()
