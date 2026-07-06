"""benchmark.metrics — Metrik hesaplama.

IoU, Precision, Recall gibi metrikleri hesaplayan modüldür.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Optional

from benchmark.data_models import BoundingBox, ValidationDataset
from benchmark.prediction_models import BenchmarkPredictions, MatchResult, DetectionPrediction

logger = logging.getLogger(__name__)


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """İki BoundingBox arasındaki Intersection over Union (IoU) değerini hesaplar."""
    inter_xtl = max(box1.xtl, box2.xtl)
    inter_ytl = max(box1.ytl, box2.ytl)
    inter_xbr = min(box1.xbr, box2.xbr)
    inter_ybr = min(box1.ybr, box2.ybr)

    if inter_xbr <= inter_xtl or inter_ybr <= inter_ytl:
        return 0.0

    inter_area = (inter_xbr - inter_xtl) * (inter_ybr - inter_ytl)
    union_area = box1.area + box2.area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


class MetricsCalculator:
    """Benchmark sonuclarini degerlendiren metrik hesaplayici."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.high_iou_threshold = 0.75

    def evaluate(
        self,
        dataset: ValidationDataset,
        predictions: BenchmarkPredictions,
    ) -> Dict[str, Any]:
        """Ground truth ile tahminleri karsilastirir."""
        
        # Build mapping from (set_name, filename) to GT boxes
        gt_map = {}
        for ref_set in dataset.reference_sets:
            for ann in ref_set.annotations:
                gt_map[(ref_set.name, ann.filename)] = ann.boxes

        global_stats = self._init_stats()
        per_set_metrics = {}

        for sp in predictions.set_predictions:
            set_stats = self._init_stats()
            
            for pred in sp.predictions:
                boxes = gt_map.get((sp.set_name, pred.image_filename), [])
                outcome, iou = self._evaluate_single(pred, boxes)
                
                # Update set stats
                set_stats[outcome] += 1
                if iou is not None:
                    set_stats["iou_sum"] += iou
                    set_stats["iou_count"] += 1
                
                # Update global stats
                global_stats[outcome] += 1
                if iou is not None:
                    global_stats["iou_sum"] += iou
                    global_stats["iou_count"] += 1

            per_set_metrics[sp.set_name] = self._compute_derived_metrics(set_stats)
            
        global_metrics = self._compute_derived_metrics(global_stats)
        
        return {
            "global": global_metrics,
            "per_set": per_set_metrics,
        }
        
    def _init_stats(self) -> Dict[str, float]:
        return {
            "True Positive": 0,
            "True Positive (>0.75)": 0,
            "False Positive": 0,
            "False Negative": 0,
            "False Negative (Error)": 0,
            "True Negative": 0,
            "iou_sum": 0.0,
            "iou_count": 0,
        }
        
    def _evaluate_single(self, pred: DetectionPrediction, gt_boxes: List[BoundingBox]) -> Tuple[str, Optional[float]]:
        if not gt_boxes:
            if pred.is_success:
                return "False Positive", 0.0
            elif pred.is_error:
                return "False Negative (Error)", None
            else:
                return "True Negative", None
                
        if pred.is_error:
            return "False Negative (Error)", None
            
        if pred.is_no_match:
            return "False Negative", None
            
        # Success and GT exists
        best_iou = 0.0
        if pred.predicted_box:
            best_iou = max((compute_iou(pred.predicted_box, gt) for gt in gt_boxes), default=0.0)
            
        if best_iou >= self.high_iou_threshold:
            return "True Positive (>0.75)", best_iou
        elif best_iou >= self.iou_threshold:
            return "True Positive", best_iou
        else:
            # Found a box but it doesn't overlap enough with GT
            return "False Positive", best_iou
            
    def _compute_derived_metrics(self, stats: Dict[str, float]) -> Dict[str, Any]:
        tp_total = stats["True Positive"] + stats["True Positive (>0.75)"]
        fp = stats["False Positive"]
        fn_total = stats["False Negative"] + stats["False Negative (Error)"]
        tn = stats["True Negative"]
        
        precision = tp_total / (tp_total + fp) if (tp_total + fp) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = stats["iou_sum"] / stats["iou_count"] if stats["iou_count"] > 0 else 0.0
        accuracy = (tp_total + tn) / (tp_total + tn + fp + fn_total) if (tp_total + tn + fp + fn_total) > 0 else 0.0
        
        return {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Accuracy": accuracy,
            "Mean IoU (Successes)": mean_iou,
            "Counts": {
                "TP": stats["True Positive"],
                "TP (>0.75)": stats["True Positive (>0.75)"],
                "FP": stats["False Positive"],
                "FN": stats["False Negative"],
                "FN (Error)": stats["False Negative (Error)"],
                "TN": stats["True Negative"],
            }
        }
