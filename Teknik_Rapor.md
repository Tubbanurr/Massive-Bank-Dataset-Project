# 🏦 Teknik Rapor: Massive Bank Transaction Big Data Pipeline

**Hazırlayan:** Proje Ekibi  
**Tarih:** 13 Mayıs 2026  
**Konu:** 1 Milyon+ Satırlık Banka Verisi Üzerinde Uçtan Uca Tahminleme ve Görselleştirme Hattı

---

## 1. Giriş ve Proje Amacı
Bu çalışma, modern finans dünyasında üretilen devasa miktardaki işlem verisinin (Big Data) gerçek zamanlı ve batch olarak işlenmesini, analiz edilmesini ve makine öğrenmesi yöntemleriyle gelecek işlem değerlerinin tahmin edilmesini amaçlamaktadır. Proje kapsamında 1,004,480 satırlık gerçekçi bir banka işlem seti kullanılmıştır.

## 2. Proje Mimarisi
Proje, yüksek ölçeklenebilirlik ve modülerlik sağlamak amacıyla mikroservis mimarisi (Docker) üzerine kurulmuştur:

- **Veri Kaynağı (Producer):** Kafka Producer, ham veriyi gerçek zamanlı bir akış (stream) olarak Kafka topic'lerine iletir.
- **Veri İşleme Katmanı (Spark Streaming & Batch):** 
    - **Spark Streaming:** Kafka'dan gelen verileri anlık olarak tüketir ve Delta Lake (Silver katmanı) üzerine yazar.
    - **Spark Batch:** EDA ve ML eğitimleri için tüm veri setini yüksek performansla işler.
- **Depolama (Delta Lake):** Veriler Bronze, Silver ve Gold katmanlarında ACID standartlarına uygun olarak saklanır.
- **Deney Takibi (MLflow):** Tüm model eğitim süreçleri, parametreler ve metrikler merkezi bir MLflow sunucusunda loglanır.
- **Sunum Katmanı (Streamlit Dashboard):** Analiz sonuçları ve model tahminleri kullanıcı dostu, interaktif bir arayüz ile sunulur.

## 3. Özellik Mühendisliği (Feature Engineering)
Model başarısını artırmak için veriden 10 adet türetilmiş özellik (feature) üretilmiştir:
- **Zaman Odaklı Özellikler:** Ay, gün, haftanın günü ve haftasonu flag'leri (mevsimselliği yakalamak için).
- **İstatistiksel Özellikler:** Lokasyon ve domain bazlı hareketli ortalamalar (Window Functions).
- **Normalizasyon:** Veri dağılımını dengelemek için Z-Score normalizasyonu uygulanmıştır.

## 4. Makine Öğrenmesi ve Performans Değerlendirmesi
Proje kapsamında 5 farklı regresyon algoritması test edilmiştir:
1. **Linear Regression:** Temel performans ölçütü (Baseline).
2. **Decision Tree Regressor:** Doğrusal olmayan ilişkilerin keşfi.
3. **Random Forest Regressor:** Topluluk öğrenmesi (Ensemble learning) ile varyansın azaltılması.
4. **Gradient Boosted Trees (GBT) Regressor:** **En yüksek performans (%89.2 R² Score).**
5. **Generalized Linear Regression (GLR):** Alternatif istatistiksel yaklaşım.

### Model Karşılaştırma Tablosu:
| Model | R² Score | RMSE (Hata) |
| :--- | :---: | :---: |
| **GBT Regressor** | **0.892** | **85,863** |
| Decision Tree | 0.873 | 93,252 |
| Random Forest | 0.834 | 106,600 |
| Linear Regression | 0.805 | 115,431 |

## 5. Karşılaşılan Zorluklar ve Çözümler
- **Veri Yapısı Tutarlılığı:** Big data hattında JSON formatlarındaki küçük değişikliklerin dashboard tarafında `AttributeError` hatalarına yol açtığı gözlemlenmiştir. Bu durum, veri yükleme fonksiyonlarına "Ultra-Robust" (hata toleranslı) mantıklar eklenerek çözülmüştür.
- **Bellek Yönetimi:** 1 milyon satırın işlenmesi sırasında Spark hafıza (OOM) hatalarını önlemek için veri bölümleme (partitioning) ve shuffle partition ayarları optimize edilmiştir.
- **Docker Ağ Yapılandırması:** Farklı konteynerlerin (Spark, Kafka, MLflow) birbirleriyle güvenli iletişimi için özel bir `internal-network` tanımlanmış ve host-bind sorunları giderilmiştir.

## 6. Sonuç ve Gelecek Çalışmalar
Proje, big data teknolojilerinin finansal tahminleme süreçlerinde ne kadar kritik bir rol oynadığını kanıtlamıştır. **GBT Regressor** modeli, karmaşık banka verileri üzerinde en güvenilir sonucu vermiştir. Gelecekte bu hatta "Real-time Drift Detection" (Veri kayması tespiti) mekanizmaları eklenerek modelin canlı veriye uyumu artırılabilir.

---
*Bu rapor, teknik mimari ve analitik sonuçların bir özetidir.*
