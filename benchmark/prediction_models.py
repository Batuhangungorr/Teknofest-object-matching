"""
benchmark.prediction_models
============================

Algoritma çıktısını temsil eden veri yapıları.

Bu modül ground truth'tan (data_models) tamamen bağımsızdır.
Ground truth ile tek ortak nokta BoundingBox sınıfıdır — çünkü
bir kutunun koordinat yapısı evrenseldir ve yeniden tanımlamak
gereksiz tekrara yol açar.

Hiyerarşi:
    MatchResult          →  algoritma sonuç durumu (enum)
    DetectionPrediction  →  tek bir görüntü için tahmin
    SetPredictions       →  bir refXX seti için tüm tahminler
    BenchmarkPredictions →  tüm setler için tüm tahminler
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from benchmark.data_models import BoundingBox


class MatchResult(Enum):
    """Eşleme algoritmasının bir test görseli için döndürdüğü sonuç durumu.

    Üç farklı durumu açıkça ayırt eder:
        SUCCESS  : Algoritma bir eşleşme buldu, bbox ve confidence mevcut.
        NO_MATCH : Algoritma çalıştı ama nesneyi bulamadı.
        ERROR    : Çalışma sırasında beklenmeyen bir hata oluştu.

    Bu ayrım, gelecekte metrik hesaplamalarında "bulunamadı" ile
    "hata" durumlarının farklı ele alınmasını sağlar.
    """

    SUCCESS = auto()
    NO_MATCH = auto()
    ERROR = auto()


@dataclass
class DetectionPrediction:
    """Tek bir test görseli için algoritmanın ürettiği tahmin.

    Attributes:
        image_filename: Test görselinin dosya adı (ör. "frame_024_00_23.jpg").
        result: Eşleme sonucunun durumu.
        predicted_box: Tahmin edilen bounding box (yalnızca SUCCESS durumunda).
        confidence: Algoritmanın güven skoru [0.0, 1.0].
                    NO_MATCH ve ERROR durumlarında 0.0 olmalıdır.
        error_message: ERROR durumunda hata açıklaması.
        metadata: Algoritma-spesifik ek bilgiler.
                  Örneğin feature extraction süresi, eşleşme sayısı vb.
    """

    image_filename: str
    result: MatchResult
    predicted_box: Optional[BoundingBox] = None
    confidence: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Tahmin başarılı mı?"""
        return self.result == MatchResult.SUCCESS

    @property
    def is_no_match(self) -> bool:
        """Algoritma nesneyi bulamadı mı?"""
        return self.result == MatchResult.NO_MATCH

    @property
    def is_error(self) -> bool:
        """Çalışma sırasında hata oluştu mu?"""
        return self.result == MatchResult.ERROR

    def __repr__(self) -> str:
        if self.is_success:
            box_repr = (
                f"({self.predicted_box.xtl:.1f}, {self.predicted_box.ytl:.1f})"
                f"->({self.predicted_box.xbr:.1f}, {self.predicted_box.ybr:.1f})"
                if self.predicted_box
                else "None"
            )
            return (
                f"DetectionPrediction({self.image_filename!r}, "
                f"SUCCESS, conf={self.confidence:.3f}, box={box_repr})"
            )
        return (
            f"DetectionPrediction({self.image_filename!r}, "
            f"{self.result.name})"
        )


@dataclass
class SetPredictions:
    """Bir referans seti (refXX) için tüm tahminleri tutan konteyner.

    Attributes:
        set_name: Referans set adı (ör. "ref01").
        engine_name: Tahminleri üreten algoritmanın adı.
        predictions: Bu setteki her test görseli için tahmin listesi.
    """

    set_name: str
    engine_name: str
    predictions: List[DetectionPrediction] = field(default_factory=list)

    @property
    def num_predictions(self) -> int:
        """Toplam tahmin sayısı."""
        return len(self.predictions)

    @property
    def num_successful(self) -> int:
        """Başarılı tahmin sayısı."""
        return sum(1 for p in self.predictions if p.is_success)

    @property
    def num_no_match(self) -> int:
        """Eşleşme bulunamayan tahmin sayısı."""
        return sum(1 for p in self.predictions if p.is_no_match)

    @property
    def num_errors(self) -> int:
        """Hata durumundaki tahmin sayısı."""
        return sum(1 for p in self.predictions if p.is_error)

    def summary(self) -> str:
        """Set tahminlerinin özet istatistikleri."""
        return (
            f"{self.set_name}: "
            f"{self.num_predictions} tahmin "
            f"(basarili={self.num_successful}, "
            f"bulunamadi={self.num_no_match}, "
            f"hata={self.num_errors})"
        )

    def __repr__(self) -> str:
        return (
            f"SetPredictions(set={self.set_name!r}, "
            f"engine={self.engine_name!r}, "
            f"predictions={self.num_predictions})"
        )


@dataclass
class BenchmarkPredictions:
    """Tüm setler icin tüm tahminleri tutan üst düzey konteyner.

    Bir benchmark çalıştırmasının tam çıktısıdır.

    Attributes:
        engine_name: Tahminleri üreten algoritmanın adı.
        set_predictions: Her referans seti için tahmin koleksiyonu.
    """

    engine_name: str
    set_predictions: List[SetPredictions] = field(default_factory=list)

    @property
    def num_sets(self) -> int:
        """Toplam set sayısı."""
        return len(self.set_predictions)

    @property
    def total_predictions(self) -> int:
        """Tüm setlerdeki toplam tahmin sayısı."""
        return sum(sp.num_predictions for sp in self.set_predictions)

    @property
    def total_successful(self) -> int:
        """Tüm setlerdeki başarılı tahmin sayısı."""
        return sum(sp.num_successful for sp in self.set_predictions)

    def get_set(self, set_name: str) -> Optional[SetPredictions]:
        """İsme göre set tahminlerini döndürür.

        Args:
            set_name: Aranacak set adı (ör. "ref01").

        Returns:
            Bulunan SetPredictions veya None.
        """
        for sp in self.set_predictions:
            if sp.set_name == set_name:
                return sp
        return None

    def summary(self) -> str:
        """Benchmark çıktısının özet istatistikleri."""
        lines = [
            f"BenchmarkPredictions: engine={self.engine_name!r}",
            f"  Toplam set       : {self.num_sets}",
            f"  Toplam tahmin    : {self.total_predictions}",
            f"  Basarili         : {self.total_successful}",
            "  -----------------------------------",
        ]
        for sp in self.set_predictions:
            lines.append(f"  {sp.summary()}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BenchmarkPredictions(engine={self.engine_name!r}, "
            f"sets={self.num_sets}, "
            f"predictions={self.total_predictions})"
        )
