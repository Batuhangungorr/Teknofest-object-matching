"""benchmark.engines.coarse_to_fine.engine — CoarseToFineEngine.

Iki asamali pipeline'i orkestre eden ana engine sinifi.
``MatchingEngine`` arayuzunu implemente eder.

TODO:
    - [ ] initialize(): DINOv2 + LightGlue model yukleme
    - [ ] set_reference(): Referans feature extraction
    - [ ] detect(): Coarse lokalizasyon -> Fine matching -> bbox
    - [ ] cleanup(): GPU bellek temizleme
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from benchmark.engine import MatchingEngine
from benchmark.prediction_models import DetectionPrediction


class CoarseToFineEngine(MatchingEngine):
    """DINOv2 + LightGlue tabanli iki asamali esleme motoru.

    Coarse asama (DINOv2):
        Test gorselindeki patch embedding'lerini referans ile
        karsilastirarak aday bolgeleri belirler.

    Fine asama (LightGlue):
        Aday bolgeler uzerinde local feature matching yaparak
        hassas bbox hesaplar.

    TODO:
        Tum metotlar implemente edilecek.
    """

    @property
    def name(self) -> str:
        return "CoarseToFine(DINOv2+LightGlue)"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        # TODO: DINOv2 model yukle
        # TODO: LightGlue (SuperPoint) model yukle
        # TODO: GPU/device ayarla
        # TODO: Config parametrelerini isle
        raise NotImplementedError("CoarseToFineEngine.initialize()")

    def set_reference(self, reference_image_path: Path) -> None:
        # TODO: Referans gorselden DINOv2 patch embedding'leri cikar
        # TODO: Referans gorselden SuperPoint keypoint'leri cikar
        # TODO: Bellekte sakla
        raise NotImplementedError("CoarseToFineEngine.set_reference()")

    def detect(self, test_image_path: Path) -> DetectionPrediction:
        # TODO: Coarse — DINOv2 heatmap -> aday bolge crop
        # TODO: Fine   — LightGlue keypoint esleme
        # TODO: Geometrik dogrulama (RANSAC)
        # TODO: Bbox hesapla
        # TODO: DetectionPrediction dondur
        raise NotImplementedError("CoarseToFineEngine.detect()")

    def cleanup(self) -> None:
        # TODO: GPU bellegini serbest birak
        # TODO: Modelleri kaldir
        # TODO: Gecici verileri temizle
        raise NotImplementedError("CoarseToFineEngine.cleanup()")
