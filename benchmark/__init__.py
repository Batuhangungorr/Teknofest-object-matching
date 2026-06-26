"""
Benchmark – Nesne Tespiti Doğrulama Altyapısı
=============================================

Bu paket, validation setleri üzerinde nesne tespiti pipeline'larını
değerlendirmek için kullanılan benchmark araçlarını içerir.

Modüller:
    data_models       : Ground truth veri yapıları (dataclass tanımları)
    parsers           : CVAT XML 1.1 anotasyon dosyası okuyucu
    loader            : Validation setlerini tarama ve yükleme
    prediction_models : Algoritma çıktısı veri yapıları
    engine            : Soyut MatchingEngine arayüzü (Strategy Pattern)
    runner            : Benchmark çalıştırma orkestratörü
"""
