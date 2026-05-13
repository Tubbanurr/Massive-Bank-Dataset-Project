"""
============================================================================
ML Training - 5 Regresyon Modeli Karsilastirmasi + MLflow
============================================================================
PROBLEM TURU: REGRESYON
TARGET (Hedef): Value (gunluk toplam islem tutari)

Gerekcesi:
- Value surekli (continuous) bir degisken → Regresyon
- Is sorusu: Domain, lokasyon, tarih ve islem adedine gore
  toplam islem tutarini tahmin et

ONEMLI - Feature Leakage onleme:
- value_zscore      → CIKARILDI (value'dan turetildi, leak)
- value_per_transaction → CIKARILDI (value/txn_count, leak)
- location_avg_value    → CIKARILDI (ayni datasetten hesaplandi)
- domain_avg_value      → CIKARILDI (ayni datasetten hesaplandi)
- value_to_location_ratio → CIKARILDI (leak)

Gercek Ozellikler (bagimsiz degiskenler):
- transaction_count, domain_encoded, event_type_encoded
- month, day_of_week, day_of_month, quarter, is_weekend

5 Model:
  1. Linear Regression
  2. Decision Tree Regressor
  3. Random Forest Regressor
  4. Gradient Boosted Trees (GBT) Regressor
  5. Generalized Linear Regression (GLR)

Metrikler: RMSE, MAE, R2, Residual Analizi
Feature Importance analizi
MLflow ile deney takibi
============================================================================
"""

import os
import json
import logging
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, month, dayofweek, dayofmonth, quarter, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor,
    GeneralizedLinearRegression
)
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import mlflow.spark

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('MLTraining')

# ===================== KONFIGURASYON =====================
CSV_FILE_PATH = os.environ.get('CSV_FILE_PATH', '/app/dataset/bankdataset.csv')
DELTA_LAKE_PATH = os.environ.get('DELTA_LAKE_PATH', '/app/delta-lake')
FEATURE_PATH = f"{DELTA_LAKE_PATH}/gold/features"
OUTPUT_PATH = '/app/output/ml'
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'file:///app/mlruns')

# ===================== FEATURE SET (LEAKAGE YOK) =====================
# NOT: value_zscore, value_per_transaction, location_avg_value,
#      domain_avg_value, value_to_location_ratio CIKARILDI (data leakage)
FEATURE_COLUMNS = [
    "transaction_count",    # Islem adedi - bagimsiz degisken
    "domain_encoded",       # Hangi alan (RETAIL, MEDICAL vs)
    "event_type_encoded",   # Islem tipi (DEPOSIT, WITHDRAWAL vs)
    "month",                # Ay (mevsimsellik)
    "day_of_week",          # Haftanin gunu
    "day_of_month",         # Ayin gunu
    "quarter",              # Ceyrek
    "is_weekend",           # Hafta sonu mu
]
LABEL_COLUMN = "value"


def create_spark_session():
    spark = SparkSession.builder \
        .appName("BankMLTraining") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.memory.fraction", "0.7") \
        .config("spark.memory.storageFraction", "0.3") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_and_prepare_features(spark):
    """
    Feature tablosundan veya CSV'den yukle.
    Leakage iceren sutunlari disla.
    """
    logger.info("=" * 60)
    logger.info("VERi YUKLEME VE HAZIRLAMA")
    logger.info("=" * 60)

    # Once Delta Lake gold'dan yukle, yoksa CSV'den uret
    try:
        df = spark.read.format("delta").load(FEATURE_PATH)
        total = df.count()
        logger.info(f"Delta Lake Gold'dan yuklendi: {total:,} satir")
    except Exception:
        logger.warning("Delta Lake bulunamadi, CSV'den yukleniyor...")
        df = _load_from_csv(spark)

    logger.info(f"Toplam veri: {df.count():,} satir")
    logger.info(f"Problem: REGRESYON | Hedef: {LABEL_COLUMN}")
    logger.info(f"Ozellikler ({len(FEATURE_COLUMNS)} adet): {FEATURE_COLUMNS}")
    logger.info("Data Leakage onlendi: value_zscore, value_per_transaction, "
                "location_avg_value, domain_avg_value, value_to_location_ratio DISLANDA")
    return df


def _load_from_csv(spark):
    """CSV'den yukle ve temel ozellikleri hesapla"""
    from pyspark.sql.functions import monotonically_increasing_id, lit
    import random
    from pyspark.sql.types import StringType
    from pyspark.sql.functions import udf

    EVENT_TYPES = ['DEPOSIT', 'WITHDRAWAL', 'TRANSFER', 'PAYMENT']
    event_udf = udf(lambda: random.choice(EVENT_TYPES), StringType())

    df = spark.read.option("header", "true").option("inferSchema", "true").csv(CSV_FILE_PATH)
    df = df.withColumn("parsed_date", to_date(col("Date"), "yyyy-MM-dd")) \
           .withColumn("event_type", event_udf()) \
           .withColumnRenamed("Date", "date") \
           .withColumnRenamed("Domain", "domain") \
           .withColumnRenamed("Location", "location") \
           .withColumnRenamed("Value", "value") \
           .withColumnRenamed("Transaction_count", "transaction_count")

    # Tarihsel ozellikler
    df = df.withColumn("month", month("parsed_date")) \
           .withColumn("day_of_week", dayofweek("parsed_date")) \
           .withColumn("day_of_month", dayofmonth("parsed_date")) \
           .withColumn("quarter", quarter("parsed_date")) \
           .withColumn("is_weekend",
               when((col("day_of_week") == 1) | (col("day_of_week") == 7), 1).otherwise(0))

    # Encoding
    domain_map = {"RESTRAUNT": 0, "INVESTMENTS": 1, "RETAIL": 2,
                  "INTERNATIONAL": 3, "PUBLIC": 4, "MEDICAL": 5, "EDUCATION": 6}
    domain_expr = when(col("domain") == "RESTRAUNT", 0)
    for d, c in list(domain_map.items())[1:]:
        domain_expr = domain_expr.when(col("domain") == d, c)
    df = df.withColumn("domain_encoded", domain_expr.otherwise(-1))

    event_expr = when(col("event_type") == "DEPOSIT", 0) \
        .when(col("event_type") == "WITHDRAWAL", 1) \
        .when(col("event_type") == "TRANSFER", 2) \
        .when(col("event_type") == "PAYMENT", 3).otherwise(-1)
    df = df.withColumn("event_type_encoded", event_expr)

    return df


def prepare_ml_data(df):
    """ML icin hazirla: temizle, assemble, scale, split"""
    logger.info("=" * 60)
    logger.info("ML VERi HAZIRLAMA")
    logger.info("=" * 60)

    # Sadece gerekli sutunlari al, null temizle
    clean_df = df.select(FEATURE_COLUMNS + [LABEL_COLUMN]).dropna()
    total = clean_df.count()
    logger.info(f"Temiz veri: {total:,} satir")

    # Value istatistikleri
    from pyspark.sql.functions import avg, stddev, min as smin, max as smax
    stats = clean_df.select(
        avg(LABEL_COLUMN).alias("mean"),
        stddev(LABEL_COLUMN).alias("std"),
        smin(LABEL_COLUMN).alias("min"),
        smax(LABEL_COLUMN).alias("max")
    ).collect()[0]
    logger.info(f"Value istatistikleri:")
    logger.info(f"  Ortalama: {stats['mean']:,.0f}")
    logger.info(f"  Std Dev : {stats['std']:,.0f}")
    logger.info(f"  Min     : {stats['min']:,}")
    logger.info(f"  Max     : {stats['max']:,}")

    # Feature vector olustur
    assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features_raw",
                                handleInvalid="skip")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withStd=True, withMean=True)
    pipeline = Pipeline(stages=[assembler, scaler])
    prep_model = pipeline.fit(clean_df)
    transformed = prep_model.transform(clean_df)

    # 80/20 Train-Test split
    train_df, test_df = transformed.randomSplit([0.8, 0.2], seed=42)
    train_count = train_df.count()
    test_count = test_df.count()
    logger.info(f"Train seti: {train_count:,} satir ({train_count/total*100:.1f}%)")
    logger.info(f"Test  seti: {test_count:,} satir ({test_count/total*100:.1f}%)")

    return train_df, test_df, prep_model


def evaluate_model(trained_model, test_df, model_name):
    """RMSE, MAE, R2 hesapla"""
    predictions = trained_model.transform(test_df)

    evaluators = {
        "rmse": RegressionEvaluator(labelCol=LABEL_COLUMN, predictionCol="prediction", metricName="rmse"),
        "mae":  RegressionEvaluator(labelCol=LABEL_COLUMN, predictionCol="prediction", metricName="mae"),
        "r2":   RegressionEvaluator(labelCol=LABEL_COLUMN, predictionCol="prediction", metricName="r2"),
    }

    rmse = evaluators["rmse"].evaluate(predictions)
    mae  = evaluators["mae"].evaluate(predictions)
    r2   = evaluators["r2"].evaluate(predictions)

    logger.info(f"  📊 {model_name}:")
    logger.info(f"     RMSE = {rmse:>12,.2f}")
    logger.info(f"     MAE  = {mae:>12,.2f}")
    logger.info(f"     R²   = {r2:>12.4f}")

    return {"model_name": model_name, "rmse": rmse, "mae": mae, "r2": r2,
            "predictions": predictions}


def get_feature_importance(trained_model, model_name):
    """Feature importance cikar"""
    importance = None
    try:
        if hasattr(trained_model, 'featureImportances'):
            imp = trained_model.featureImportances.toArray()
            importance = dict(zip(FEATURE_COLUMNS, imp.tolist()))
        elif hasattr(trained_model, 'coefficients'):
            coefs = np.abs(trained_model.coefficients.toArray())
            # Normalize
            total = coefs.sum()
            if total > 0:
                coefs = coefs / total
            importance = dict(zip(FEATURE_COLUMNS, coefs.tolist()))
    except Exception as e:
        logger.warning(f"Feature importance alinamadi ({model_name}): {e}")
    return importance


def train_all_models(train_df, test_df):
    """5 regresyon modeli egit, MLflow'a logla"""
    logger.info("=" * 60)
    logger.info("MODEL EGiTiMi BASLIYOR (5 Model)")
    logger.info("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("bank_value_regression")

    # 5 Regresyon Modeli
    models = {
        "1_Linear_Regression": LinearRegression(
            featuresCol="features", labelCol=LABEL_COLUMN,
            maxIter=100, regParam=0.1, elasticNetParam=0.0
        ),
        "2_Decision_Tree": DecisionTreeRegressor(
            featuresCol="features", labelCol=LABEL_COLUMN,
            maxDepth=8, minInstancesPerNode=10
        ),
        "3_Random_Forest": RandomForestRegressor(
            featuresCol="features", labelCol=LABEL_COLUMN,
            numTrees=20, maxDepth=8, seed=42
        ),
        "4_GBT_Regressor": GBTRegressor(
            featuresCol="features", labelCol=LABEL_COLUMN,
            maxIter=20, maxDepth=5, stepSize=0.1
        ),
        "5_GLR": GeneralizedLinearRegression(
            featuresCol="features", labelCol=LABEL_COLUMN,
            family="gaussian", link="identity", maxIter=100, regParam=0.1
        )
    }

    all_results = []

    for name, model_obj in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Model egitiliyor: {name}")
        logger.info(f"{'='*50}")

        with mlflow.start_run(run_name=name):
            # Parametreleri logla
            mlflow.log_param("model_type", name)
            mlflow.log_param("problem_type", "regression")
            mlflow.log_param("target", LABEL_COLUMN)
            mlflow.log_param("features", str(FEATURE_COLUMNS))
            mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
            mlflow.log_param("data_leakage_prevented", True)

            # Egit
            trained_model = model_obj.fit(train_df)

            # Degerlendir
            result = evaluate_model(trained_model, test_df, name)
            all_results.append(result)

            # Metrikleri logla
            mlflow.log_metric("rmse", result["rmse"])
            mlflow.log_metric("mae",  result["mae"])
            mlflow.log_metric("r2",   result["r2"])

            # Feature importance
            fi = get_feature_importance(trained_model, name)
            if fi:
                result["feature_importance"] = fi
                logger.info(f"  Feature Importance ({name}):")
                sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
                for feat, imp in sorted_fi:
                    logger.info(f"    {feat:30s}: {imp:.4f}")
                    mlflow.log_metric(f"fi_{feat}", imp)

            # Modeli kaydet
            try:
                mlflow.spark.log_model(trained_model, f"model_{name}")
                logger.info(f"  ✅ Model MLflow'a kaydedildi: {name}")
            except Exception as e:
                logger.warning(f"  ⚠️ Model kaydedilemedi: {e}")

    return all_results


def create_all_plots(results, output_path):
    """
    Tum gorselleri uret:
    1. Model karsilastirma (RMSE, MAE, R2)
    2. Feature Importance (en iyi model)
    3. Gercek vs Tahmin scatter (en iyi model)
    4. Residual dagilimi (en iyi model)
    """
    os.makedirs(output_path, exist_ok=True)

    names     = [r["model_name"].replace("1_","").replace("2_","").replace("3_","")
                  .replace("4_","").replace("5_","") for r in results]
    rmse_vals = [r["rmse"] for r in results]
    mae_vals  = [r["mae"]  for r in results]
    r2_vals   = [r["r2"]   for r in results]

    colors = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6']

    # ── 1. Model Performans Karsilastirmasi ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('5 Regresyon Modeli Performans Karsilastirmasi\n'
                 f'(Veri: {sum(r["predictions"].count() for r in results[:1]):,}+ satir, Hedef: Value)',
                 fontsize=13, fontweight='bold')

    axes[0].barh(names, rmse_vals, color=colors)
    axes[0].set_title('RMSE (dusuk = iyi)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('RMSE')
    for i, v in enumerate(rmse_vals):
        axes[0].text(v, i, f' {v:,.0f}', va='center', fontsize=8)

    axes[1].barh(names, mae_vals, color=colors)
    axes[1].set_title('MAE (dusuk = iyi)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('MAE')
    for i, v in enumerate(mae_vals):
        axes[1].text(v, i, f' {v:,.0f}', va='center', fontsize=8)

    axes[2].barh(names, r2_vals, color=colors)
    axes[2].set_title('R² Score (yuksek = iyi)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('R²')
    for i, v in enumerate(r2_vals):
        axes[2].text(max(v, 0), i, f' {v:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{output_path}/01_model_comparison.png', dpi=150)
    plt.close()
    logger.info("Gorsell 1/4: Model karsilastirma kaydedildi")

    # ── 2. Feature Importance (en iyi R2 modeli) ──
    best = max(results, key=lambda x: x.get("r2", -999))
    if "feature_importance" in best:
        fi = best["feature_importance"]
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        feats, imps = zip(*sorted_fi)

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(feats, imps, color='#3498db')
        ax.set_title(f'Feature Importance\n({best["model_name"].split("_",1)[1]})',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Importance (Normalize)')
        for bar, imp in zip(bars, imps):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f' {imp:.4f}', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(f'{output_path}/02_feature_importance.png', dpi=150)
        plt.close()
        logger.info("Gorsel 2/4: Feature importance kaydedildi")

    # ── 3. Gercek vs Tahmin (en iyi model) ──
    pred_pdf = best["predictions"].select(LABEL_COLUMN, "prediction").sample(
        fraction=min(10000/best["predictions"].count(), 1.0), seed=42
    ).toPandas()

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(pred_pdf[LABEL_COLUMN], pred_pdf["prediction"],
               alpha=0.3, s=10, c='#3498db', label='Tahminler')
    mn = min(pred_pdf[LABEL_COLUMN].min(), pred_pdf["prediction"].min())
    mx = max(pred_pdf[LABEL_COLUMN].max(), pred_pdf["prediction"].max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Ideal (y=x)')
    ax.set_title(f'Gercek vs Tahmin\n({best["model_name"].split("_",1)[1]} | R²={best["r2"]:.4f})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Gercek Value'); ax.set_ylabel('Tahmin Value')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}/03_actual_vs_predicted.png', dpi=150)
    plt.close()
    logger.info("Gorsel 3/4: Gercek vs tahmin kaydedildi")

    # ── 4. Residual Analizi ──
    pred_pdf['residual'] = pred_pdf[LABEL_COLUMN] - pred_pdf["prediction"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Residual (Artik) Analizi — {best["model_name"].split("_",1)[1]}',
                 fontsize=13, fontweight='bold')

    # Histogram
    axes[0].hist(pred_pdf['residual'], bins=60, color='#e74c3c', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=2, label='Sifir')
    axes[0].set_title('Residual Dagilimi', fontsize=11)
    axes[0].set_xlabel('Residual (Gercek - Tahmin)')
    axes[0].set_ylabel('Frekans')
    axes[0].legend()

    # Scatter residual vs predicted
    axes[1].scatter(pred_pdf["prediction"], pred_pdf['residual'],
                    alpha=0.3, s=8, c='#9b59b6')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_title('Residual vs Tahmin', fontsize=11)
    axes[1].set_xlabel('Tahmin Value')
    axes[1].set_ylabel('Residual')

    plt.tight_layout()
    plt.savefig(f'{output_path}/04_residual_analysis.png', dpi=150)
    plt.close()
    logger.info("Gorsel 4/4: Residual analizi kaydedildi")


def save_results(results, output_path):
    """Sonuclari JSON olarak kaydet"""
    os.makedirs(output_path, exist_ok=True)
    summary = []
    for r in results:
        summary.append({
            "model_name": r["model_name"],
            "problem_type": "regression",
            "target": LABEL_COLUMN,
            "features": FEATURE_COLUMNS,
            "rmse": round(r["rmse"], 2),
            "mae":  round(r["mae"],  2),
            "r2":   round(r["r2"],   6),
            "feature_importance": r.get("feature_importance", {})
        })

    # R2'ye gore sirala
    summary.sort(key=lambda x: x["r2"], reverse=True)

    with open(f'{output_path}/model_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"Sonuclar JSON'a kaydedildi: {output_path}/model_results.json")
    return summary


def print_final_summary(results):
    """Final karsilastirma tablosu"""
    logger.info("\n" + "=" * 70)
    logger.info("FINAL MODEL KARSILASTIRMASI (R2'ye gore siralanmis)")
    logger.info("=" * 70)
    logger.info(f"{'Model':<30} {'RMSE':>12} {'MAE':>12} {'R2':>8}")
    logger.info("-" * 70)
    sorted_results = sorted(results, key=lambda x: x["r2"], reverse=True)
    for r in sorted_results:
        name = r["model_name"].split("_", 1)[1] if "_" in r["model_name"] else r["model_name"]
        logger.info(f"{name:<30} {r['rmse']:>12,.2f} {r['mae']:>12,.2f} {r['r2']:>8.4f}")
    logger.info("=" * 70)
    best = sorted_results[0]
    bname = best["model_name"].split("_", 1)[1]
    logger.info(f"🏆 EN IYI MODEL: {bname}")
    logger.info(f"   RMSE : {best['rmse']:,.2f}")
    logger.info(f"   MAE  : {best['mae']:,.2f}")
    logger.info(f"   R²   : {best['r2']:.4f}")


def main():
    import os as _os
    _os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

    logger.info("=" * 60)
    logger.info("ML TRAINING BASLIYOR")
    logger.info("Problem: REGRESYON | Hedef: value (islem tutari)")
    logger.info("=" * 60)

    spark = create_spark_session()
    df = load_and_prepare_features(spark)
    train_df, test_df, prep_model = prepare_ml_data(df)
    results = train_all_models(train_df, test_df)
    create_all_plots(results, OUTPUT_PATH)
    summary = save_results(results, OUTPUT_PATH)
    print_final_summary(results)

    logger.info("ML TRAINING TAMAMLANDI!")
    spark.stop()


if __name__ == '__main__':
    main()
