"""benchmark.metrics — Metrik hesaplama iskeleti.

Gelecekte IoU, Precision, Recall, mAP gibi metrikleri
hesaplayacak olan moduldur.  Su an yalnizca sinif iskeleti
ve genisletme noktalari tanimlanmistir.

Kullanim (gelecek)::

    calculator = MetricsCalculator()
    report = calculator.evaluate(dataset, predictions)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from benchmark.data_models import ValidationDataset
from benchmark.prediction_models import BenchmarkPredictions

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Benchmark sonuclarini degerlendiren metrik hesaplayici.

    Bu sinif su an bir iskelet olup, gelecekte asagidaki metriklerle
    genisletilecektir:

    - IoU (Intersection over Union)
    - Precision / Recall
    - mAP (mean Average Precision)
    - Accuracy
    - F1 Score

    Note:
        Metrik hesaplama, ground truth (``ValidationDataset``) ile
        prediction (``BenchmarkPredictions``) arasinda yapilir.
        Her iki veri yapisi da birbirinden bagimsizdir ve yalnizca
        bu sinif icerisinde bir araya getirilir.
    """

    def evaluate(
        self,
        dataset: ValidationDataset,
        predictions: BenchmarkPredictions,
    ) -> Dict[str, Any]:
        """Ground truth ile tahminleri karsilastirir.

        Args:
            dataset: Ground truth anotasyonlarini iceren validation seti.
            predictions: Algoritmanin urettigi tahminler.

        Returns:
            Metrik sonuclarini iceren sozluk.

        Raises:
            NotImplementedError: Henuz implemente edilmedi.

        TODO:
            - IoU hesaplama
            - Precision/Recall egrileri
            - mAP hesaplama
            - Confusion matrix
            - Per-set ve toplam metrikler
        """
        raise NotImplementedError(
            "MetricsCalculator.evaluate() henuz implemente edilmedi. "
            "Gelecek surumde IoU, Precision, Recall, mAP eklenecek."
        )
