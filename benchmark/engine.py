"""
benchmark.engine
=================

Tüm eşleme algoritmalarının uyması gereken soyut arayüz.

Bu modül Strategy Pattern'i uygular. Benchmark sistemi hiçbir zaman
somut bir algoritmayı (DINOv2, LightGlue, LoFTR, SuperGlue, OpenCV vb.)
bilmez — yalnızca bu arayüz üzerinden iletişim kurar.

Yaşam döngüsü:
    initialize() → N × [set_reference() → M × detect()] → cleanup()

Context manager protokolü de desteklenir:
    with engine:
        engine.set_reference(...)
        engine.detect(...)

Yeni bir algoritma eklemek için:
    1. MatchingEngine'i miras al
    2. Tüm abstract metotları implemente et
    3. BenchmarkRunner'a inject et
    4. Benchmark koduna hiçbir değişiklik gerekmez
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from benchmark.prediction_models import DetectionPrediction


class MatchingEngine(ABC):
    """Nesne eşleme algoritmalarının soyut temel sınıfı.

    Her somut engine bu sınıfı miras alarak tüm abstract metotları
    implemente etmelidir. Benchmark sistemi yalnızca bu arayüzü bilir.

    Typical usage::

        class MyEngine(MatchingEngine):
            def name(self) -> str:
                return "MyAlgorithm"
            ...

        engine = MyEngine()
        runner = BenchmarkRunner(engine=engine, dataset=dataset)
        results = runner.run()
    """

    # ------------------------------------------------------------------
    # Abstract Interface — alt sınıfların implemente etmesi zorunlu
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Algoritmanın insanlar tarafından okunabilir adı.

        Returns:
            Algoritma adı (ör. "DINOv2+LightGlue", "LoFTR", "ORB+BFMatcher").
        """

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Algoritmayı çalışmaya hazırlar.

        Model yükleme, GPU bellek tahsisi, parametre ayarlama gibi
        tek seferlik hazırlık işlemlerini burada yapın.

        Args:
            config: Algoritma-spesifik yapılandırma parametreleri.
                    Örneğin: {"device": "cuda", "threshold": 0.5}
                    None ise varsayılan ayarlar kullanılmalıdır.
        """

    @abstractmethod
    def set_reference(self, reference_image_path: Path) -> None:
        """Referans görselini işler ve belleğe alır.

        Bu metot her yeni referans nesnesi için bir kez çağrılır.
        Referans feature'larının çıkarılması ve saklanması burada yapılır.

        Performans notu:
            Referans feature'ları belleğe alınarak, aynı referans için
            birden fazla test görseli değerlendirilirken tekrar
            hesaplanmak zorunda kalmaz.

        Args:
            reference_image_path: Referans görselinin dosya yolu.

        Raises:
            FileNotFoundError: Görsel dosyası bulunamazsa.
        """

    @abstractmethod
    def detect(self, test_image_path: Path) -> DetectionPrediction:
        """Test görselinde referans nesnesini arar.

        Daha önce set_reference() ile yüklenen referans nesnesini
        test görselinde bulmaya çalışır.

        Args:
            test_image_path: Test görselinin dosya yolu.

        Returns:
            DetectionPrediction nesnesi. Üç olası durum:
                - SUCCESS  : Nesne bulundu, predicted_box ve confidence dolu.
                - NO_MATCH : Nesne bulunamadı.
                - ERROR    : Beklenmeyen hata oluştu.

        Note:
            Bu metot hiçbir zaman exception fırlatmamalıdır.
            Hatalar MatchResult.ERROR ile raporlanmalıdır.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Kaynakları serbest bırakır.

        GPU belleği, geçici dosyalar, açık bağlantılar vb.
        temizleme işlemlerini burada yapın.

        Bu metot idempotent olmalıdır — birden fazla çağrılabilir.
        """

    # ------------------------------------------------------------------
    # Context Manager Protocol — `with` bloğuyla kullanım
    # ------------------------------------------------------------------

    def __enter__(self) -> MatchingEngine:
        """Context manager girişi. Engine'i döndürür.

        Not: initialize() burada otomatik çağrılmaz çünkü config
        parametresi gerekebilir. Kullanıcı initialize()'ı açıkça
        çağırmalıdır.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager çıkışı. cleanup() çağırır."""
        self.cleanup()
        return None  # Exception'ları yutma

    # ------------------------------------------------------------------
    # Ortak yardımcı metotlar
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
