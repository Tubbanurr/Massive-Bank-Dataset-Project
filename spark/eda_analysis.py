"""
============================================================================
EDA (Exploratory Data Analysis) - Kesifsel Veri Analizi
============================================================================
Tum 1,004,480 satiri analiz etmek icin dogrudan CSV'den okur.
- Temel istatistikler
- Zaman serisi trendleri
- Kategorik ve sayisal dagilimlar
- Eksik deger analizi
============================================================================
"""

import os
import logging
import json
import random
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, year, month, dayofweek, dayofmonth, count,
    sum as spark_sum, avg, min as spark_min, max as spark_max,
    stddev, to_date, udf, lit, desc, asc,
    when, isnan, isnull
)
from pyspark.sql.types import StringType
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('EDA')

CSV_FILE_PATH = os.environ.get('CSV_FILE_PATH', '/app/dataset/bankdataset.csv')
OUTPUT_PATH = '/app/output/eda'

EVENT_TYPES = ['DEPOSIT', 'WITHDRAWAL', 'TRANSFER', 'PAYMENT']


def create_spark_session():
    spark = SparkSession.builder \
        .appName("BankDatasetEDA") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_data(spark):
    """Dogrudan CSV'den tum veriyi yukle (1,004,480 satir)"""
    logger.info(f"CSV'den veri yukleniyor: {CSV_FILE_PATH}")
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(CSV_FILE_PATH)

    # Date sutununu parse et
    df = df.withColumn("parsed_date", to_date(col("Date"), "yyyy-MM-dd"))

    # Simule event_type ekle (pipeline ile uyumlu)
    event_udf = udf(lambda: random.choice(EVENT_TYPES), StringType())
    df = df.withColumn("event_type", event_udf())

    # Sutun isimlerini kucult
    df = df.withColumnRenamed("Date", "date") \
           .withColumnRenamed("Domain", "domain") \
           .withColumnRenamed("Location", "location") \
           .withColumnRenamed("Value", "value") \
           .withColumnRenamed("Transaction_count", "transaction_count")

    total = df.count()
    logger.info(f"Toplam satir: {total:,}")
    logger.info(f"Sutunlar: {df.columns}")
    return df


def basic_statistics(df):
    """Temel istatistikler"""
    logger.info("=" * 50)
    logger.info("TEMEL ISTATISTIKLER")
    logger.info("=" * 50)

    stats = {}
    stats['total_rows'] = df.count()
    stats['total_columns'] = len(df.columns)

    numeric_stats = df.select(
        count("*").alias("count"),
        avg("value").alias("avg_value"),
        stddev("value").alias("std_value"),
        spark_min("value").alias("min_value"),
        spark_max("value").alias("max_value"),
        avg("transaction_count").alias("avg_txn_count"),
        stddev("transaction_count").alias("std_txn_count"),
        spark_min("transaction_count").alias("min_txn_count"),
        spark_max("transaction_count").alias("max_txn_count")
    ).collect()[0]

    stats['numeric'] = {
        'value': {
            'mean': float(numeric_stats['avg_value']),
            'std': float(numeric_stats['std_value']),
            'min': int(numeric_stats['min_value']),
            'max': int(numeric_stats['max_value'])
        },
        'transaction_count': {
            'mean': float(numeric_stats['avg_txn_count']),
            'std': float(numeric_stats['std_txn_count']),
            'min': int(numeric_stats['min_txn_count']),
            'max': int(numeric_stats['max_txn_count'])
        }
    }

    stats['unique_domains'] = df.select("domain").distinct().count()
    stats['unique_locations'] = df.select("location").distinct().count()
    stats['unique_event_types'] = df.select("event_type").distinct().count()

    logger.info(f"Total rows: {stats['total_rows']:,}")
    logger.info(f"Unique domains: {stats['unique_domains']}")
    logger.info(f"Unique locations: {stats['unique_locations']}")

    return stats


def null_analysis(df):
    """Eksik deger analizi"""
    logger.info("EKSIK DEGER ANALIZI")
    null_counts = {}
    for c in df.columns:
        nc = df.filter(col(c).isNull()).count()
        null_counts[c] = nc
        if nc > 0:
            logger.info(f"  {c}: {nc} null ({nc/df.count()*100:.2f}%)")
    if all(v == 0 for v in null_counts.values()):
        logger.info("  Eksik deger YOK!")
    return null_counts


def domain_analysis(df):
    """Domain bazli analiz"""
    logger.info("DOMAIN BAZLI ANALIZ")
    domain_stats = df.groupBy("domain").agg(
        count("*").alias("count"),
        spark_sum("value").alias("total_value"),
        avg("value").alias("avg_value"),
        avg("transaction_count").alias("avg_txn_count")
    ).orderBy(desc("count"))
    domain_stats.show(truncate=False)
    return domain_stats.toPandas()


def location_analysis(df):
    """Lokasyon bazli analiz - Tum lokasyonlar"""
    logger.info("LOKASYON BAZLI ANALIZ (Top 15)")
    loc_stats = df.groupBy("location").agg(
        count("*").alias("count"),
        spark_sum("value").alias("total_value"),
        avg("value").alias("avg_value")
    ).orderBy(desc("total_value")).limit(15)
    loc_stats.show(truncate=False)
    return loc_stats.toPandas()


def time_series_analysis(df):
    """Zaman serisi analizi - Tum aylar"""
    logger.info("ZAMAN SERISI ANALIZI (Aylik)")
    monthly = df.withColumn("year", year("parsed_date")) \
        .withColumn("month", month("parsed_date")) \
        .groupBy("year", "month").agg(
            count("*").alias("count"),
            spark_sum("value").alias("total_value"),
            avg("value").alias("avg_value")
        ).orderBy("year", "month")
    monthly.show(truncate=False)
    return monthly.toPandas()


def create_visualizations(df, domain_df, location_df, monthly_df, output_path):
    """EDA gorsellestirmeleri - 1M satir uzerinden"""
    os.makedirs(output_path, exist_ok=True)

    # Gorseller icin pandas'a cevir (scatter icin sample al, digerleri tam veri)
    pdf_sample = df.select("value", "transaction_count", "domain", "location",
                            "event_type", "parsed_date").sample(fraction=0.05, seed=42).toPandas()
    pdf_full_domain = df.select("value", "domain").toPandas()

    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. Value dagilimi (histogram)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pdf_full_domain['value'], bins=80, color='#3498db', edgecolor='black', alpha=0.7)
    ax.set_title(f'Islem Degeri Dagilimi (Value) — {len(pdf_full_domain):,} satir', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value'); ax.set_ylabel('Frekans')
    plt.tight_layout()
    plt.savefig(f'{output_path}/01_value_distribution.png', dpi=150)
    plt.close()

    # 2. Transaction count dagilimi
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pdf_sample['transaction_count'], bins=60, color='#2ecc71', edgecolor='black', alpha=0.7)
    ax.set_title('Islem Sayisi Dagilimi', fontsize=14, fontweight='bold')
    ax.set_xlabel('Transaction Count'); ax.set_ylabel('Frekans')
    plt.tight_layout()
    plt.savefig(f'{output_path}/02_txn_count_distribution.png', dpi=150)
    plt.close()

    # 3. Domain bazli pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22']
    ax.pie(domain_df['count'], labels=domain_df['domain'], autopct='%1.1f%%',
           colors=colors[:len(domain_df)], startangle=90)
    ax.set_title('Domain Dagilimi', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_path}/03_domain_pie.png', dpi=150)
    plt.close()

    # 4. Domain bazli ortalama deger (bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(domain_df['domain'], domain_df['avg_value'], color='#3498db')
    ax.set_title('Domain Bazli Ortalama Islem Degeri', fontsize=14, fontweight='bold')
    ax.set_xlabel('Ortalama Value')
    plt.tight_layout()
    plt.savefig(f'{output_path}/04_domain_avg_value.png', dpi=150)
    plt.close()

    # 5. Top 15 Lokasyon (bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(location_df['location'], location_df['total_value'], color='#9b59b6')
    ax.set_title('Top 15 Lokasyon (Toplam Deger)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Total Value')
    plt.tight_layout()
    plt.savefig(f'{output_path}/05_top_locations.png', dpi=150)
    plt.close()

    # 6. Aylik trend (line chart)
    fig, ax = plt.subplots(figsize=(14, 6))
    monthly_df['month_str'] = monthly_df['month'].astype(str)
    ax.plot(monthly_df['month_str'], monthly_df['total_value'], 'o-', color='#e74c3c', linewidth=2, markersize=8)
    for i, row in monthly_df.iterrows():
        ax.annotate(f"{row['count']:,}", (str(row['month']), row['total_value']),
                    textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)
    ax.set_title('Aylik Toplam Islem Degeri Trendi (2022)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Ay'); ax.set_ylabel('Toplam Value')
    plt.tight_layout()
    plt.savefig(f'{output_path}/06_monthly_trend.png', dpi=150)
    plt.close()

    # 7. Value vs Transaction Count scatter (sample)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(pdf_sample['transaction_count'], pdf_sample['value'], alpha=0.3, s=8, c='#3498db')
    ax.set_title(f'Value vs Transaction Count (sample: {len(pdf_sample):,})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Transaction Count'); ax.set_ylabel('Value')
    plt.tight_layout()
    plt.savefig(f'{output_path}/07_value_vs_txn.png', dpi=150)
    plt.close()

    # 8. Event type dagilimi
    fig, ax = plt.subplots(figsize=(8, 6))
    evt = pdf_sample['event_type'].value_counts()
    ax.bar(evt.index, evt.values, color=['#e74c3c','#3498db','#2ecc71','#f39c12'])
    ax.set_title('Event Type Dagilimi', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frekans')
    plt.tight_layout()
    plt.savefig(f'{output_path}/08_event_type_dist.png', dpi=150)
    plt.close()

    # 9. Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = pdf_full_domain[['value']].join(
        df.select("transaction_count").sample(fraction=0.05, seed=42).toPandas()
    ).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    ax.set_title('Korelasyon Matrisi', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_path}/09_correlation.png', dpi=150)
    plt.close()

    # 10. Boxplot - Domain bazli Value (sample)
    fig, ax = plt.subplots(figsize=(14, 6))
    pdf_full_domain.boxplot(column='value', by='domain', ax=ax)
    ax.set_title('Domain Bazli Value Boxplot', fontsize=14, fontweight='bold')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(f'{output_path}/10_domain_boxplot.png', dpi=150)
    plt.close()

    logger.info(f"10 gorsel {output_path} klasorune kaydedildi")


def main():
    logger.info("=" * 60)
    logger.info("EDA (KESIFSEL VERI ANALIZI) BASLIYOR — TAM VERI SETI")
    logger.info("=" * 60)
    spark = create_spark_session()
    df = load_data(spark)

    stats = basic_statistics(df)
    null_counts = null_analysis(df)
    domain_df = domain_analysis(df)
    location_df = location_analysis(df)
    monthly_df = time_series_analysis(df)

    create_visualizations(df, domain_df, location_df, monthly_df, OUTPUT_PATH)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with open(f'{OUTPUT_PATH}/eda_results.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    domain_df.to_csv(f'{OUTPUT_PATH}/domain_stats.csv', index=False)
    location_df.to_csv(f'{OUTPUT_PATH}/location_stats.csv', index=False)
    monthly_df.to_csv(f'{OUTPUT_PATH}/monthly_stats.csv', index=False)

    logger.info("EDA TAMAMLANDI!")
    spark.stop()


if __name__ == '__main__':
    main()
