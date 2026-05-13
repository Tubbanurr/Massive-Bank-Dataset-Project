"""
============================================================================
Feature Engineering - Ozellik Muhendisligi
============================================================================
Silver katmanindan okunan veriye yeni ozellikler ekler ve
Delta Lake Gold katmanina feature tablosu olarak kaydeder.

Uretilen Ozellikler (en az 5):
1. value_per_transaction: Islem basina deger (Value/Transaction_count)
   -> Is mantigi: Her bir islemin ortalama degerini gosterir
2. month / day_of_week / day_of_month: Tarihsel ozellikler
   -> Is mantigi: Mevsimsel ve haftalik trendleri yakalar
3. is_weekend: Hafta sonu mu? (1/0)
   -> Is mantigi: Hafta sonu islem davranislarini ayirt eder
4. value_category: Deger kategorisi (LOW/MEDIUM/HIGH/VERY_HIGH)
   -> Is mantigi: Islem buyuklugune gore segmentasyon
5. domain_encoded: Domain numerik encoding
   -> Is mantigi: ML modelleri icin kategorik degiskeni sayisal yapar
6. location_avg_value: Lokasyon bazli ortalama deger
   -> Is mantigi: Lokasyonun genel islem duzeyini gosterir
7. domain_avg_value: Domain bazli ortalama deger
   -> Is mantigi: Domain'in ortalama islem buyuklugunu gosterir
8. value_zscore: Z-score (standartlastirilmis deger)
   -> Is mantigi: Anomali tespiti icin kullanilir
============================================================================
"""

import os
import logging
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, year, month, dayofweek, dayofmonth, quarter,
    when, lit, avg, stddev, count, dense_rank,
    round as spark_round
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('FeatureEngineering')

DELTA_LAKE_PATH = os.environ.get('DELTA_LAKE_PATH', '/app/delta-lake')
SILVER_PATH = f"{DELTA_LAKE_PATH}/silver/bank_transactions"
FEATURE_PATH = f"{DELTA_LAKE_PATH}/gold/features"


def create_spark_session():
    spark = SparkSession.builder \
        .appName("BankFeatureEngineering") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_silver_data(spark):
    df = spark.read.format("delta").load(SILVER_PATH)
    logger.info(f"Silver data loaded: {df.count()} rows")
    return df


def add_features(df):
    """Tum ozellikleri ekle"""
    logger.info("Feature engineering basliyor...")

    # 1. value_per_transaction
    df = df.withColumn(
        "value_per_transaction",
        spark_round(col("value") / col("transaction_count"), 2)
    )
    logger.info("  [1] value_per_transaction eklendi")

    # 2. Tarihsel ozellikler
    df = df.withColumn("month", month("parsed_date")) \
        .withColumn("day_of_week", dayofweek("parsed_date")) \
        .withColumn("day_of_month", dayofmonth("parsed_date")) \
        .withColumn("quarter", quarter("parsed_date"))
    logger.info("  [2] Tarihsel ozellikler eklendi (month, day_of_week, day_of_month, quarter)")

    # 3. is_weekend
    df = df.withColumn(
        "is_weekend",
        when((col("day_of_week") == 1) | (col("day_of_week") == 7), 1).otherwise(0)
    )
    logger.info("  [3] is_weekend eklendi")

    # 4. value_category
    df = df.withColumn(
        "value_category",
        when(col("value") < 400000, "LOW")
        .when(col("value") < 700000, "MEDIUM")
        .when(col("value") < 1000000, "HIGH")
        .otherwise("VERY_HIGH")
    )
    logger.info("  [4] value_category eklendi")

    # 5. domain_encoded (StringIndexer benzeri)
    domain_map = {
        "RESTRAUNT": 0, "INVESTMENTS": 1, "RETAIL": 2,
        "INTERNATIONAL": 3, "PUBLIC": 4, "MEDICAL": 5, "EDUCATION": 6
    }
    domain_expr = when(col("domain") == "RESTRAUNT", 0)
    for domain, code in list(domain_map.items())[1:]:
        domain_expr = domain_expr.when(col("domain") == domain, code)
    df = df.withColumn("domain_encoded", domain_expr.otherwise(-1))
    logger.info("  [5] domain_encoded eklendi")

    # 6. event_type_encoded
    event_map = {"DEPOSIT": 0, "WITHDRAWAL": 1, "TRANSFER": 2, "PAYMENT": 3}
    event_expr = when(col("event_type") == "DEPOSIT", 0)
    for evt, code in list(event_map.items())[1:]:
        event_expr = event_expr.when(col("event_type") == evt, code)
    df = df.withColumn("event_type_encoded", event_expr.otherwise(-1))
    logger.info("  [6] event_type_encoded eklendi")

    # 7. location_avg_value (window function)
    loc_window = Window.partitionBy("location")
    df = df.withColumn("location_avg_value", spark_round(avg("value").over(loc_window), 2))
    logger.info("  [7] location_avg_value eklendi")

    # 8. domain_avg_value (window function)
    dom_window = Window.partitionBy("domain")
    df = df.withColumn("domain_avg_value", spark_round(avg("value").over(dom_window), 2))
    logger.info("  [8] domain_avg_value eklendi")

    # 9. value_zscore
    global_stats = df.select(avg("value").alias("mean_v"), stddev("value").alias("std_v")).collect()[0]
    mean_val = global_stats["mean_v"]
    std_val = global_stats["std_v"]
    df = df.withColumn(
        "value_zscore",
        spark_round((col("value") - lit(mean_val)) / lit(std_val), 4)
    )
    logger.info("  [9] value_zscore eklendi")

    # 10. value_to_location_ratio
    df = df.withColumn(
        "value_to_location_ratio",
        spark_round(col("value") / col("location_avg_value"), 4)
    )
    logger.info("  [10] value_to_location_ratio eklendi")

    return df


def save_features(df):
    """Feature tablosunu Delta Lake'e kaydet"""
    logger.info(f"Feature tablosu kaydediliyor: {FEATURE_PATH}")

    # Sadece gerekli sutunlari sec
    feature_cols = [
        "message_id", "date", "domain", "location", "event_type",
        "value", "transaction_count",
        # Yeni ozellikler
        "value_per_transaction", "month", "day_of_week", "day_of_month",
        "quarter", "is_weekend", "value_category",
        "domain_encoded", "event_type_encoded",
        "location_avg_value", "domain_avg_value",
        "value_zscore", "value_to_location_ratio"
    ]

    feature_df = df.select(*feature_cols)
    feature_df.write.format("delta").mode("overwrite").save(FEATURE_PATH)

    logger.info(f"Feature tablosu kaydedildi: {feature_df.count()} satir, {len(feature_cols)} sutun")
    feature_df.printSchema()
    feature_df.show(5, truncate=False)
    return feature_df


def main():
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING BASLIYOR")
    logger.info("=" * 60)
    spark = create_spark_session()
    df = load_silver_data(spark)
    df = add_features(df)
    save_features(df)
    logger.info("FEATURE ENGINEERING TAMAMLANDI!")
    spark.stop()


if __name__ == '__main__':
    main()
