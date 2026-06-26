"""
benchmark.runner
=================

Benchmark çalıştırma orkestratörü.

Bu modül veri yükleme katmanı ile eşleme motoru arasındaki
köprüyü kurar. Kendisi hiçbir algoritmayı bilmez ve hiçbir
metrik hesaplamaz.

Sorumlulukları:
    1. Validation setlerini dolaşmak
    2. MatchingEngine'i doğru sırayla çağırmak
    3. DetectionPrediction nesnelerini toplamak
    4. BenchmarkPredictions döndürmek

Kullanım:
    from benchmark.loader import load_validation_dataset
    from benchmark.runner import BenchmarkRunner

    dataset = load_validation_dataset("Validation")
    engine = SomeMatchingEngine()  # MatchingEngine implementasyonu
    runner = BenchmarkRunner(engine=engine, dataset=dataset)
    results = runner.run(config={"threshold": 0.5})
    print(results.summary())
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from benchmark.data_models import ReferenceSet, ValidationDataset
from benchmark.engine import MatchingEngine
from benchmark.prediction_models import (
    BenchmarkPredictions,
    DetectionPrediction,
    MatchResult,
    SetPredictions,
)

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Benchmark çalıştırma orkestratörü.

    Dependency Injection prensibiyle çalışır: engine ve dataset
    dışarıdan verilir, runner yalnızca akışı yönetir.

    Attributes:
        engine: Eşleme motoru (MatchingEngine implementasyonu).
        dataset: Yüklenmiş validation veri seti.
    """

    def __init__(
        self,
        engine: MatchingEngine,
        dataset: ValidationDataset,
    ) -> None:
        """BenchmarkRunner oluşturur.

        Args:
            engine: Kullanılacak eşleme motoru.
            dataset: Üzerinde benchmark çalıştırılacak validation seti.

        Raises:
            TypeError: engine MatchingEngine değilse.
            ValueError: dataset boşsa.
        """
        if not isinstance(engine, MatchingEngine):
            raise TypeError(
                f"engine bir MatchingEngine instance'ı olmalı, "
                f"verilen: {type(engine).__name__}"
            )
        if not dataset.reference_sets:
            raise ValueError(
                "dataset boş — en az bir referans seti gerekli."
            )

        self._engine = engine
        self._dataset = dataset

    @property
    def engine(self) -> MatchingEngine:
        """Kullanılan eşleme motoru."""
        return self._engine

    @property
    def dataset(self) -> ValidationDataset:
        """Üzerinde çalışılan validation seti."""
        return self._dataset

    def run(
        self, config: Optional[Dict[str, Any]] = None
    ) -> BenchmarkPredictions:
        """Tüm referans setleri üzerinde benchmark'ı çalıştırır.

        Akış:
            1. engine.initialize(config)
            2. Her referans seti için:
               a. engine.set_reference(referans_görsel)
               b. Her test görseli için engine.detect(test_görsel)
               c. Tahminleri topla
            3. engine.cleanup()
            4. BenchmarkPredictions döndür

        Args:
            config: Engine'e iletilecek yapılandırma parametreleri.

        Returns:
            Tüm setlerin tahminlerini içeren BenchmarkPredictions.
        """
        engine_name = self._engine.name

        logger.info(
            "Benchmark baslatiliyor: engine=%r, sets=%d, images=%d",
            engine_name,
            self._dataset.num_sets,
            self._dataset.total_images,
        )

        total_start = time.perf_counter()

        # 1. Engine'i hazırla
        logger.info("Engine hazırlaniyor: %r", engine_name)
        self._engine.initialize(config)

        # 2. Her set üzerinde çalıştır
        all_set_predictions: List[SetPredictions] = []

        for ref_set in self._dataset.reference_sets:
            set_predictions = self._run_single_set(ref_set)
            all_set_predictions.append(set_predictions)

        # 3. Engine'i temizle
        logger.info("Engine temizleniyor: %r", engine_name)
        self._engine.cleanup()

        # 4. Sonuçları paketle
        total_elapsed = time.perf_counter() - total_start

        results = BenchmarkPredictions(
            engine_name=engine_name,
            set_predictions=all_set_predictions,
        )

        logger.info(
            "Benchmark tamamlandi: %d set, %d tahmin, %.2f saniye",
            results.num_sets,
            results.total_predictions,
            total_elapsed,
        )

        return results

    def _run_single_set(self, ref_set: ReferenceSet) -> SetPredictions:
        """Tek bir referans seti üzerinde engine'i çalıştırır.

        Args:
            ref_set: İşlenecek referans seti.

        Returns:
            Bu setin tüm tahminlerini içeren SetPredictions.
        """
        engine_name = self._engine.name

        logger.info(
            "Set calistiriliyor: %s (%d goruntu)",
            ref_set.name,
            ref_set.num_images,
        )

        set_start = time.perf_counter()

        # Referans görseli yükle
        if ref_set.reference_image_path is None:
            logger.warning(
                "%s: Referans gorseli bulunamadi, set atlaniyor.",
                ref_set.name,
            )
            return self._create_skipped_set(ref_set, engine_name)

        logger.info(
            "%s: Referans gorseli yukleniyor: %s",
            ref_set.name,
            ref_set.reference_image_path.name,
        )
        self._engine.set_reference(ref_set.reference_image_path)

        # Her test görseli için detect çağır
        predictions: List[DetectionPrediction] = []

        for annotation in ref_set.annotations:
            prediction = self._detect_single_image(
                ref_set.name, annotation.filepath, annotation.filename
            )
            predictions.append(prediction)

        set_elapsed = time.perf_counter() - set_start

        set_preds = SetPredictions(
            set_name=ref_set.name,
            engine_name=engine_name,
            predictions=predictions,
        )

        logger.info(
            "%s tamamlandi: %d tahmin "
            "(basarili=%d, bulunamadi=%d, hata=%d) %.2f sn",
            ref_set.name,
            set_preds.num_predictions,
            set_preds.num_successful,
            set_preds.num_no_match,
            set_preds.num_errors,
            set_elapsed,
        )

        return set_preds

    def _detect_single_image(
        self,
        set_name: str,
        image_path: "Path",
        image_filename: str,
    ) -> DetectionPrediction:
        """Tek bir test görseli için detect çağrısını güvenli şekilde yapar.

        Engine.detect() hata fırlatmamalıdır ama savunmacı programlama
        olarak yakalanmayan exception'ları burada ele alıyoruz.

        Args:
            set_name: Log mesajları için set adı.
            image_path: Test görselinin tam yolu.
            image_filename: Test görselinin dosya adı.

        Returns:
            DetectionPrediction nesnesi.
        """
        try:
            prediction = self._engine.detect(image_path)
            return prediction
        except Exception as e:
            logger.error(
                "%s/%s: Engine exception: %s",
                set_name,
                image_filename,
                e,
            )
            return DetectionPrediction(
                image_filename=image_filename,
                result=MatchResult.ERROR,
                confidence=0.0,
                error_message=f"Engine exception: {e}",
            )

    @staticmethod
    def _create_skipped_set(
        ref_set: ReferenceSet, engine_name: str
    ) -> SetPredictions:
        """Referans görseli olmayan bir set için boş prediction oluşturur.

        Args:
            ref_set: Atlanan referans seti.
            engine_name: Algoritma adı.

        Returns:
            Tüm tahminleri ERROR olan SetPredictions.
        """
        predictions = [
            DetectionPrediction(
                image_filename=ann.filename,
                result=MatchResult.ERROR,
                confidence=0.0,
                error_message="Referans gorseli bulunamadi.",
            )
            for ann in ref_set.annotations
        ]

        return SetPredictions(
            set_name=ref_set.name,
            engine_name=engine_name,
            predictions=predictions,
        )
