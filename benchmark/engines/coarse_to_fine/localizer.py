"""benchmark.engines.coarse_to_fine.localizer — Coarse lokalizasyon.

DINOv2 patch embedding'lerinden benzerlik heatmap'i olusturarak
aday bolgeleri belirler.

TODO:
    - [ ] Referans ve test patch embedding'leri arasinda cosine similarity
    - [ ] Heatmap olusturma ve threshold uygulama
    - [ ] Aday bolge (ROI) cikarma
    - [ ] Multi-scale heatmap destegi
"""

from __future__ import annotations


class CoarseLocalizer:
    """Heatmap tabanli coarse lokalizasyon bileseni.

    Sorumluluklar:
        - Referans-test patch similarity hesaplama.
        - Benzerlik heatmap'i olusturma.
        - Aday bolge (crop koordinatlari) cikarma.

    TODO:
        Tum metotlar implemente edilecek.
    """

    def compute_heatmap(
        self,
        reference_features: object,
        test_features: object,
    ) -> None:
        """Referans ve test feature'lari arasinda benzerlik heatmap'i olusturur.

        Args:
            reference_features: Referans patch embedding'leri.
            test_features: Test patch embedding'leri.

        Returns:
            2D benzerlik heatmap'i.

        TODO:
            - Cosine similarity hesaplama
            - Heatmap normalizasyonu
            - Gaussian smoothing (opsiyonel)
        """
        raise NotImplementedError

    def extract_candidate_regions(self, heatmap: object) -> None:
        """Heatmap'ten aday bolgeleri cikarir.

        Args:
            heatmap: Benzerlik heatmap'i.

        Returns:
            Aday bolge koordinatlari listesi.

        TODO:
            - Threshold uygulama
            - Connected component analizi
            - NMS (Non-Maximum Suppression)
            - Minimum alan filtreleme
        """
        raise NotImplementedError
