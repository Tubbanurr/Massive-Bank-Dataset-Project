# 🏦 Massive Bank Dataset - Büyük Veri Pipeline Projesi

Gerçek dünya senaryosuna uygun, uçtan uca büyük veri pipeline'ı. Docker ile konteynerize edilmiş ortamda Apache Kafka, Apache Spark, Delta Lake, MLflow ve Streamlit teknolojilerini kullanır.

## 📊 Veri Seti

| Özellik | Değer |
|---------|-------|
| **Satır sayısı** | 1,004,480 |
| **Sütunlar** | Date, Domain, Location, Value, Transaction_count |
| **Tarih aralığı** | 2022-01-01 → 2022-12-31 |
| **Domain'ler** | RESTRAUNT, INVESTMENTS, RETAIL, INTERNATIONAL, PUBLIC, MEDICAL, EDUCATION |
| **Lokasyonlar** | 46 farklı şehir |

## 🏗️ Mimari

```
CSV Dataset → Kafka Producer → Kafka Broker → Spark Structured Streaming
                                                      ↓
                                                Bronze Layer (Raw)
                                                      ↓
                                                Silver Layer (Clean)
                                                      ↓
                                                Gold Layer (Features)
                                                      ↓
                                            ML Models + MLflow Tracking
                                                      ↓
                                              Streamlit Dashboard
```

## 🛠️ Teknoloji Yığını

| Katman | Teknoloji | Açıklama |
|--------|-----------|----------|
| Konteynerizasyon | Docker, Docker Compose | Tüm servislerin izole ortamda çalıştırılması |
| Veri Üretimi | Apache Kafka + Python Producer | Gerçek zamanlı veri akışı simülasyonu |
| Veri İşleme | Apache Spark (PySpark) | Batch ve streaming veri işleme, ETL pipeline |
| Depolama | Delta Lake | ACID uyumlu, versiyonlanabilir veri deposu |
| Makine Öğrenmesi | Spark MLlib + MLflow | Model eğitimi, karşılaştırma ve deney takibi |
| Görselleştirme | Streamlit + Plotly | Sonuçların görsel olarak sunulması |

## 🚀 Kurulum ve Çalıştırma

### Ön Koşullar
- Docker & Docker Compose kurulu olmalı
- En az 8GB RAM önerilir

### 1. Projeyi Başlatma

```bash
# Xlsx'i CSV'ye çevir (eğer henüz yapılmadıysa)
python3 -c "import pandas as pd; df=pd.read_excel('dataset/bankdataset.xlsx'); df.to_csv('dataset/bankdataset.csv', index=False)"

# Tüm servisleri başlat
docker-compose up -d --build

# Logları takip et
docker-compose logs -f
```

### 2. Pipeline Adımları

```bash
# Adım 1: Kafka Producer otomatik başlar ve CSV'yi Kafka'ya stream eder
docker-compose logs -f producer

# Adım 2: Spark Streaming otomatik başlar ve Delta Lake'e yazar
docker-compose logs -f spark-streaming

# Adım 3: EDA analizi çalıştır
docker-compose exec spark-master /opt/spark/bin/spark-submit \
  --packages io.delta:delta-spark_2.12:3.2.0 \
  --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
  --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
  /app/spark/eda_analysis.py

# Adım 4: Feature Engineering
docker-compose exec spark-master /opt/spark/bin/spark-submit \
  --packages io.delta:delta-spark_2.12:3.2.0 \
  --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
  --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
  /app/spark/feature_engineering.py

# Adım 5: ML Training
docker-compose exec spark-master /opt/spark/bin/spark-submit \
  --packages io.delta:delta-spark_2.12:3.2.0 \
  --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
  --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
  /app/spark/ml_training.py
```

### 3. Web Arayüzleri

| Servis | URL | Açıklama |
|--------|-----|----------|
| **Spark Master UI** | http://localhost:8080 | Spark cluster durumu |
| **Spark Worker UI** | http://localhost:8081 | Worker durumu |
| **Spark Application UI** | http://localhost:4040 | Çalışan uygulama detayları |
| **MLflow UI** | http://localhost:5000 | Deney takibi ve model karşılaştırma |
| **Dashboard** | http://localhost:8501 | Streamlit interaktif dashboard |

### 4. Durdurma

```bash
docker-compose down
# Tüm verileri silerek durdurma:
docker-compose down -v
```

## 📁 Proje Yapısı

```
Massive-Bank-Dataset-Project/
├── docker-compose.yml           # Tüm servislerin tanımı
├── Dockerfile.producer          # Kafka Producer image
├── Dockerfile.spark             # Spark + Delta Lake image
├── dataset/
│   ├── bankdataset.xlsx         # Orijinal veri seti
│   └── bankdataset.csv          # CSV formatı (producer için)
├── producer/
│   └── kafka_producer.py        # CSV → Kafka streaming
├── spark/
│   ├── spark_streaming.py       # Kafka → Delta Lake (Bronze/Silver/Gold)
│   ├── eda_analysis.py          # Keşifsel Veri Analizi
│   ├── feature_engineering.py   # Özellik mühendisliği (10 yeni özellik)
│   └── ml_training.py           # 5 model eğitimi + MLflow
├── dashboard/
│   └── dashboard.py             # Streamlit dashboard (5 sayfa)
├── delta-lake/                  # Delta Lake veri deposu (otomatik oluşur)
├── mlruns/                      # MLflow deney verileri (otomatik oluşur)
├── output/                      # Görseller ve sonuçlar (otomatik oluşur)
└── README.md
```

## 📊 Proje Adımları

### Adım 1: Docker Ortamı
- Zookeeper, Kafka, Spark Master/Worker, MLflow, Dashboard servisleri
- Health check'ler ile servis bağımlılıkları yönetimi

### Adım 2: Kafka ile Streaming Veri Üretimi
- CSV satırları JSON formatında Kafka'ya gönderilir
- Her mesajda: timestamp, user_id, event_type, orijinal veriler
- Ayarlanabilir hız (MESSAGE_RATE env variable, 10-100 mesaj/sn)
- Domain bazlı partitioning

### Adım 3: Spark Structured Streaming
- **Bronze**: Ham veri olduğu gibi Delta Lake'e
- **Silver**: Null temizleme, duplike kaldırma, format doğrulama
- **Gold**: Domain/Location bazlı aggregation

### Adım 4: Keşifsel Veri Analizi (EDA)
- Temel istatistikler (ortalama, std, min, max)
- Domain ve lokasyon bazlı analizler
- Zaman serisi trendleri (aylık)
- 10 farklı görselleştirme

### Adım 5: Feature Engineering (10 Özellik)
1. **value_per_transaction**: İşlem başına değer
2. **month/day_of_week/day_of_month/quarter**: Tarihsel özellikler
3. **is_weekend**: Hafta sonu flag'i
4. **value_category**: Değer kategorisi (LOW/MEDIUM/HIGH/VERY_HIGH)
5. **domain_encoded**: Domain numerik encoding
6. **event_type_encoded**: Event type numerik encoding
7. **location_avg_value**: Lokasyon ortalama değeri
8. **domain_avg_value**: Domain ortalama değeri
9. **value_zscore**: Standartlaştırılmış değer
10. **value_to_location_ratio**: Lokasyon ortalamasına oran

### Adım 6: ML Model Karşılaştırma (Regression)
| # | Model | Açıklama |
|---|-------|----------|
| 1 | Linear Regression | Temel doğrusal model |
| 2 | Decision Tree Regressor | Karar ağacı |
| 3 | Random Forest Regressor | Ensemble ağaç modeli |
| 4 | GBT Regressor | Gradient boosting |
| 5 | GLR | Genelleştirilmiş doğrusal regresyon |

**Metrikler**: RMSE, MAE, R² Score
**Feature Importance**: Tüm modeller için
**MLflow**: Parametre, metrik ve model loglama

### Adım 7: Dashboard
- 5 sayfalı Streamlit dashboard
- Plotly ile interaktif grafikler
- Model performans karşılaştırma (grouped bar chart)
- Feature importance (horizontal bar chart)
- Gerçek vs Tahmin scatter plot
- Residual dağılım grafiği
- EDA görselleştirmeleri