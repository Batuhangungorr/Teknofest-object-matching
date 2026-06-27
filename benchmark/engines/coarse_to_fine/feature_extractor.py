"""benchmark.engines.coarse_to_fine.feature_extractor — DINOv2 wrapper.

DINOv2 ViT modelini sarmalayarak patch-level ve global
feature extraction islemlerini gerceklestirir.

TODO:
    - [ ] DINOv2 model yukleme (ViT-S/14 veya ViT-B/14)
    - [ ] Gorsel on-isleme (resize, normalize)
    - [ ] Patch token extraction
    - [ ] Global CLS token extraction
    - [ ] TensorRT / ONNX optimizasyonu (Jetson icin)
"""

from __future__ import annotations


class FeatureExtractor:
    """DINOv2 tabanli feature extraction bileşeni.

    Sorumluluklar:
        - Gorselden DINOv2 patch embedding'leri cikarma.
        - Global CLS token cikarma.
        - Model yasam dongusu yonetimi.

    TODO:
        Tum metotlar implemente edilecek.
    """

    def load_model(self, model_name: str = "dinov2_vits14") -> None:
        """DINOv2 modelini yukler.

        Args:
            model_name: Kullanilacak model adi.

        TODO:
            - torch.hub ile model yukleme
            - Device (cuda/cpu) yonetimi
            - Model eval moduna alma
        """
        raise NotImplementedError

    def extract_patch_features(self, image_path: "Path") -> None:
        """Gorselden patch-level feature'lar cikarir.

        Args:
            image_path: Gorsel dosya yolu.

        Returns:
            Patch embedding tensoru.

        TODO:
            - Gorsel yukleme ve on-isleme
            - ViT forward pass
            - Patch token'larini dondurme
        """
        raise NotImplementedError

    def extract_global_features(self, image_path: "Path") -> None:
        """Gorselden global CLS token cikarir.

        Args:
            image_path: Gorsel dosya yolu.

        Returns:
            Global feature vektoru.

        TODO:
            - Gorsel yukleme
            - CLS token extraction
        """
        raise NotImplementedError

    def cleanup(self) -> None:
        """Model ve GPU kaynaklarini serbest birakir.

        TODO:
            - Model referansini sil
            - GPU cache temizle
        """
        raise NotImplementedError
