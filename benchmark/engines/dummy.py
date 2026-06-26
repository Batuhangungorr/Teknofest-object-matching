"""
benchmark.engines.dummy
========================

Pipeline doğrulama amaçlı sahte eşleme motoru.

Bu motor gerçek görüntü işleme yapmaz. Amacı benchmark akışının
(loader → runner → engine → prediction) baştan sona doğru
çalıştığını kanıtlamaktır.

Her test görseli için sabit koordinatlarda sahte bir bounding box
ve rastgele bir güven skoru üretir.

Kullanım:
    from benchmark.engines.dummy import DummyMatchingEngine
    engine = DummyMatchingEngine()
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

from benchmark.data_models import BoundingBox
from benchmark.engine import MatchingEngine
from benchmark.prediction_models import DetectionPrediction, MatchResult

logger = logging.getLogger(__name__)


class DummyMatchingEngine(MatchingEngine):
    """Pipeline doğrulama amaçlı sahte eşleme motoru.

    Gerçek bir eşleme algoritması değildir. Benchmark altyapısının
    uçtan uca çalıştığını doğrulamak için kullanılır.

    Davranış:
        - initialize: config'den random seed alır (tekrarlanabilirlik).
        - set_reference: Dosya varlığını kontrol eder, başka işlem yapmaz.
        - detect: Sabit bir bbox ve rastgele confidence üretir.
                  Yapılandırılabilir hata oranı ile NO_MATCH ve ERROR
                  durumlarını simüle edebilir.
        - cleanup: Dahili durumu sıfırlar.

    Config parametreleri:
        seed (int): Random seed. Varsayılan: 42.
        no_match_rate (float): NO_MATCH döndürme olasılığı [0, 1].
                               Varsayılan: 0.1.
        error_rate (float): ERROR döndürme olasılığı [0, 1].
                            Varsayılan: 0.05.
    """

    def __init__(self) -> None:
        self._rng: Optional[random.Random] = None
        self._reference_path: Optional[Path] = None
        self._no_match_rate: float = 0.1
        self._error_rate: float = 0.05
        self._initialized: bool = False

    @property
    def name(self) -> str:
        return "DummyMatchingEngine"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}

        seed = config.get("seed", 42)
        self._rng = random.Random(seed)
        self._no_match_rate = config.get("no_match_rate", 0.1)
        self._error_rate = config.get("error_rate", 0.05)
        self._initialized = True

        logger.info(
            "%s baslatildi: seed=%s, no_match_rate=%.2f, error_rate=%.2f",
            self.name,
            seed,
            self._no_match_rate,
            self._error_rate,
        )

    def set_reference(self, reference_image_path: Path) -> None:
        if not self._initialized:
            raise RuntimeError("Engine henuz initialize edilmedi.")

        reference_image_path = Path(reference_image_path)

        if not reference_image_path.is_file():
            raise FileNotFoundError(
                f"Referans gorseli bulunamadi: {reference_image_path}"
            )

        self._reference_path = reference_image_path
        logger.info(
            "%s: Referans ayarlandi: %s",
            self.name,
            reference_image_path.name,
        )

    def detect(self, test_image_path: Path) -> DetectionPrediction:
        if not self._initialized or self._rng is None:
            raise RuntimeError("Engine henuz initialize edilmedi.")

        test_image_path = Path(test_image_path)
        filename = test_image_path.name

        # Rastgele sonuç durumu simüle et
        roll = self._rng.random()

        # ERROR durumu
        if roll < self._error_rate:
            logger.debug("%s: ERROR simule edildi: %s", self.name, filename)
            return DetectionPrediction(
                image_filename=filename,
                result=MatchResult.ERROR,
                confidence=0.0,
                error_message="Simulated error for pipeline testing.",
                metadata={"simulated": True},
            )

        # NO_MATCH durumu
        if roll < self._error_rate + self._no_match_rate:
            logger.debug(
                "%s: NO_MATCH simule edildi: %s", self.name, filename
            )
            return DetectionPrediction(
                image_filename=filename,
                result=MatchResult.NO_MATCH,
                confidence=0.0,
                metadata={"simulated": True},
            )

        # SUCCESS durumu — sabit sahte bbox + rastgele confidence
        confidence = round(self._rng.uniform(0.5, 1.0), 4)

        dummy_box = BoundingBox(
            xtl=100.0,
            ytl=100.0,
            xbr=300.0,
            ybr=300.0,
            label="target",
            occluded=False,
        )

        logger.debug(
            "%s: SUCCESS simule edildi: %s (conf=%.4f)",
            self.name,
            filename,
            confidence,
        )

        return DetectionPrediction(
            image_filename=filename,
            result=MatchResult.SUCCESS,
            predicted_box=dummy_box,
            confidence=confidence,
            metadata={"simulated": True},
        )

    def cleanup(self) -> None:
        self._rng = None
        self._reference_path = None
        self._initialized = False
        logger.info("%s: Temizlendi.", self.name)
