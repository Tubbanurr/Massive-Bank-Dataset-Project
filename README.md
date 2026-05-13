# 🏦 Massive Bank Transaction Big Data Pipeline

![Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Kafka](https://img.shields.io/badge/Apache_Kafka-231F20?style=for-the-badge&logo=apachekafka&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)

Bu proje, **1 Milyon+ satırlık** devasa banka işlem verisini gerçek zamanlı ve batch olarak işleyen, gelişmiş makine öğrenmesi (ML) modelleriyle işlem değerlerini tahmin eden ve interaktif bir dashboard üzerinden sunan uçtan uca (end-to-end) bir big data mimarisidir.

---

## 🚀 Öne Çıkan Özellikler
- **Gerçek Zamanlı Veri Akışı:** Kafka üzerinden simüle edilen banka işlemlerinin Spark Streaming ile anlık işlenmesi.
- **Büyük Veri İşleme:** Apache Spark (PySpark) kullanılarak yüksek performanslı Feature Engineering ve EDA.
- **Gelişmiş Makine Öğrenmesi:** 5 farklı regresyon modelinin (Linear, Decision Tree, Random Forest, GBT, GLR) eğitilmesi ve karşılaştırılması.
- **Interaktif Dashboard:** Streamlit tabanlı, derinlemesine EDA ve ML analizleri sunan görsel arayüz.
- **Deney Takibi:** Tüm model parametreleri ve metriklerinin MLflow ile loglanması.

---

## 📊 Model Performans Sonuçları (Regresyon Analizi)

Eğitilen modeller arasında **Gradient Boosted Trees (GBT) Regressor** en yüksek performansı göstermiştir.

| Model | R² Score | RMSE (Hata) | MAE (Hata) |
| :--- | :---: | :---: | :---: |
| **GBT Regressor** | **0.892** | **85,863** | **67,082** |
| Decision Tree | 0.873 | 93,252 | 72,063 |
| Random Forest | 0.834 | 106,600 | 87,332 |
| Linear Regression | 0.805 | 115,431 | 97,116 |
| GLR | 0.805 | 115,431 | 97,116 |

---

## 🔍 Dashboard İçeriği
Proje Dashboard'u (`dashboard.py`) akademik ve teknik gereklilikleri %100 karşılayan şu görselleştirmeleri içerir:

### 1. EDA Analizi (Keşifsel Veri Analizi)
- **Domain Dağılımı (Pie Chart):** İşlemlerin hangi platformlardan geldiğinin yüzdesel dağılımı.
- **İşlem Değeri Dağılımı (Histogram):** Para transferi miktarlarının yoğunluk analizi.
- **Lokasyon Bazlı Analiz:** En çok işlem yapılan şehirlerin toplam işlem hacmi.

### 2. ML Model Karşılaştırması
- **Grouped Bar Chart:** 5 modelin RMSE ve MAE değerlerinin yan yana kıyaslanması.
- **Feature Importance (Horizontal Bar Chart):** Tahmini en çok etkileyen değişkenlerin (İşlem yoğunluğu, High Value vb.) sıralaması.
- **Regresyon Görselleri:** 
    - **Scatter Plot:** Gerçek vs Tahmin değerlerinin korelasyonu.
    - **Residual (Artık) Dağılımı:** Model hatalarının normal dağılım analizi.

### 3. Zaman Serisi ve Trend Analizi
- **31 Günlük Trend:** Günlük toplam işlem hacmi ve işlem sayısındaki periyodik dalgalanmaların (haftalık döngüler) line-chart üzerinden analizi.

---

## 🛠 Teknik Mimari
Proje tamamen **Dockerize** edilmiştir ve şu bileşenleri içerir:
- **Spark Master/Worker:** Veri işleme ve ML eğitimi için.
- **Kafka & Zookeeper:** Veri akışı simülasyonu için.
- **MLflow Server:** Deney takibi ve model kaydı için.
- **Streamlit:** Son kullanıcı arayüzü için.

## ⚙️ Kurulum ve Çalıştırma
```bash
# 1. Konteynırları ayağa kaldırın
docker-compose up -d --build

# 2. Producer'ı başlatın (Veri üretimi)
python producer/kafka_producer.py

# 3. Spark ML Eğitimini başlatın
docker exec -it spark-master spark-submit /app/spark/ml_training.py
```

---
*Bu çalışma, Büyük Veri Analitiği prensiplerine uygun olarak, yüksek performanslı tahminleme ve görselleştirme hedefleriyle geliştirilmiştir.*