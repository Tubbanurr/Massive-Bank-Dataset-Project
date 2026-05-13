"""
============================================================================
ML Training - 5 Model Karsilastirmasi + MLflow
============================================================================
Regression problemi: Value tahmini
5 Model:
  1. Linear Regression
  2. Decision Tree Regressor
  3. Random Forest Regressor
  4. Gradient Boosted Trees (GBT) Regressor
  5. Generalized Linear Regression (GLR)

Metrikler: RMSE, MAE, R2
Feature Importance analizi
MLflow ile deney takibi
============================================================================
"""

import os
import json
import logging
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
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

DELTA_LAKE_PATH = os.environ.get('DELTA_LAKE_PATH', '/app/delta-lake')
FEATURE_PATH = f"{DELTA_LAKE_PATH}/gold/features"
OUTPUT_PATH = '/app/output/ml'
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'file:///app/mlruns')

# Ozellik sutunlari
FEATURE_COLUMNS = [
    "transaction_count", "value_per_transaction", "month",
    "day_of_week", "day_of_month", "quarter", "is_weekend",
    "domain_encoded", "event_type_encoded",
    "location_avg_value", "domain_avg_value", "value_zscore"
]
LABEL_COLUMN = "value"


def create_spark_session():
    spark = SparkSession.builder \
        .appName("BankMLTraining") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_features(spark):
    df = spark.read.format("delta").load(FEATURE_PATH)
    logger.info(f"Feature data loaded: {df.count()} rows")
    return df


def prepare_data(df):
    """Veriyi ML icin hazirla"""
    # Null olanlari cikar
    clean_df = df.select(FEATURE_COLUMNS + [LABEL_COLUMN]).dropna()
    logger.info(f"Clean data: {clean_df.count()} rows")

    # Vector assembler
    assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withStd=True, withMean=True)

    pipeline = Pipeline(stages=[assembler, scaler])
    model = pipeline.fit(clean_df)
    transformed = model.transform(clean_df)

    # Train/Test split (80/20)
    train_df, test_df = transformed.randomSplit([0.8, 0.2], seed=42)
    logger.info(f"Train: {train_df.count()}, Test: {test_df.count()}")
    return train_df, test_df, model


def evaluate_model(model, test_df, model_name):
    """Model degerlendir"""
    predictions = model.transform(test_df)

    evaluator_rmse = RegressionEvaluator(
        labelCol=LABEL_COLUMN, predictionCol="prediction", metricName="rmse")
    evaluator_mae = RegressionEvaluator(
        labelCol=LABEL_COLUMN, predictionCol="prediction", metricName="mae")
    evaluator_r2 = RegressionEvaluator(
        labelCol=LABEL_COLUMN, predictionCol="prediction", metricName="r2")

    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    logger.info(f"  {model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")
    return {
        'model_name': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions
    }


def get_feature_importance(model, model_name, feature_names):
    """Feature importance cikar"""
    importance = None
    try:
        if hasattr(model, 'featureImportances'):
            imp = model.featureImportances.toArray()
            importance = dict(zip(feature_names, imp.tolist()))
        elif hasattr(model, 'coefficients'):
            coefs = model.coefficients.toArray()
            importance = dict(zip(feature_names, np.abs(coefs).tolist()))
    except Exception as e:
        logger.warning(f"Feature importance alinamadi ({model_name}): {e}")
    return importance


def train_all_models(train_df, test_df):
    """5 modeli egit ve karsilastir"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("bank_transaction_regression")

    models = {
        "Linear_Regression": LinearRegression(
            featuresCol="features", labelCol=LABEL_COLUMN, maxIter=100
        ),
        "Decision_Tree": DecisionTreeRegressor(
            featuresCol="features", labelCol=LABEL_COLUMN, maxDepth=10
        ),
        "Random_Forest": RandomForestRegressor(
            featuresCol="features", labelCol=LABEL_COLUMN, numTrees=50, maxDepth=10
        ),
        "GBT_Regressor": GBTRegressor(
            featuresCol="features", labelCol=LABEL_COLUMN, maxIter=50, maxDepth=5
        ),
        "GLR": GeneralizedLinearRegression(
            featuresCol="features", labelCol=LABEL_COLUMN,
            family="gaussian", link="identity", maxIter=100
        )
    }

    all_results = []

    for name, model_obj in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Model egitiliyor: {name}")
        logger.info(f"{'='*50}")

        with mlflow.start_run(run_name=name):
            # Egit
            trained_model = model_obj.fit(train_df)

            # Degerlendir
            result = evaluate_model(trained_model, test_df, name)
            all_results.append(result)

            # MLflow loglama
            mlflow.log_param("model_type", name)
            mlflow.log_param("features", str(FEATURE_COLUMNS))
            mlflow.log_param("label", LABEL_COLUMN)
            mlflow.log_metric("rmse", result['rmse'])
            mlflow.log_metric("mae", result['mae'])
            mlflow.log_metric("r2", result['r2'])

            # Feature importance
            fi = get_feature_importance(trained_model, name, FEATURE_COLUMNS)
            if fi:
                result['feature_importance'] = fi
                for feat, imp in fi.items():
                    mlflow.log_metric(f"fi_{feat}", imp)

            # Model kaydet
            mlflow.spark.log_model(trained_model, f"model_{name}")
            logger.info(f"Model MLflow'a kaydedildi: {name}")

    return all_results


def create_comparison_plots(results, output_path):
    """Model karsilastirma gorselleri"""
    os.makedirs(output_path, exist_ok=True)

    names = [r['model_name'] for r in results]
    rmse_vals = [r['rmse'] for r in results]
    mae_vals = [r['mae'] for r in results]
    r2_vals = [r['r2'] for r in results]

    # 1. Model performans karsilastirma (grouped bar chart)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    axes[0].barh(names, rmse_vals, color=colors)
    axes[0].set_title('RMSE (dusuk=iyi)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('RMSE')

    axes[1].barh(names, mae_vals, color=colors)
    axes[1].set_title('MAE (dusuk=iyi)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('MAE')

    axes[2].barh(names, r2_vals, color=colors)
    axes[2].set_title('R2 Score (yuksek=iyi)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('R2')

    plt.suptitle('5 Model Performans Karsilastirmasi', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_path}/model_comparison.png', dpi=150)
    plt.close()

    # 2. Feature Importance (en iyi model)
    best_model = max(results, key=lambda x: x.get('r2', 0))
    if 'feature_importance' in best_model:
        fi = best_model['feature_importance']
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        feats, imps = zip(*sorted_fi)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(feats, imps, color='#3498db')
        ax.set_title(f'Feature Importance ({best_model["model_name"]})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'{output_path}/feature_importance.png', dpi=150)
        plt.close()

    # 3. Gercek vs Tahmin scatter (en iyi model)
    pred_pdf = best_model['predictions'].select(
        LABEL_COLUMN, "prediction").toPandas().sample(min(5000, len(best_model['predictions'].toPandas())))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pred_pdf[LABEL_COLUMN], pred_pdf['prediction'], alpha=0.3, s=10, c='#3498db')
    mn = min(pred_pdf[LABEL_COLUMN].min(), pred_pdf['prediction'].min())
    mx = max(pred_pdf[LABEL_COLUMN].max(), pred_pdf['prediction'].max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Ideal')
    ax.set_title(f'Gercek vs Tahmin ({best_model["model_name"]})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Gercek Value'); ax.set_ylabel('Tahmin Value')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}/actual_vs_predicted.png', dpi=150)
    plt.close()

    # 4. Residual dagilimi
    pred_pdf['residual'] = pred_pdf[LABEL_COLUMN] - pred_pdf['prediction']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pred_pdf['residual'], bins=50, color='#e74c3c', edgecolor='black', alpha=0.7)
    ax.set_title(f'Residual Dagilimi ({best_model["model_name"]})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Residual (Gercek - Tahmin)'); ax.set_ylabel('Frekans')
    ax.axvline(x=0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_path}/residual_distribution.png', dpi=150)
    plt.close()

    logger.info(f"Gorseller kaydedildi: {output_path}")


def save_results(results, output_path):
    """Sonuclari JSON olarak kaydet"""
    os.makedirs(output_path, exist_ok=True)
    summary = []
    for r in results:
        summary.append({
            'model_name': r['model_name'],
            'rmse': r['rmse'],
            'mae': r['mae'],
            'r2': r['r2'],
            'feature_importance': r.get('feature_importance', {})
        })

    with open(f'{output_path}/model_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Sonuclar kaydedildi: {output_path}/model_results.json")


def main():
    logger.info("=" * 60)
    logger.info("ML TRAINING BASLIYOR")
    logger.info("=" * 60)
    spark = create_spark_session()
    df = load_features(spark)
    train_df, test_df, prep_model = prepare_data(df)
    results = train_all_models(train_df, test_df)
    create_comparison_plots(results, OUTPUT_PATH)
    save_results(results, OUTPUT_PATH)

    # En iyi model
    best = max(results, key=lambda x: x['r2'])
    logger.info(f"\nEN IYI MODEL: {best['model_name']}")
    logger.info(f"  RMSE: {best['rmse']:.2f}")
    logger.info(f"  MAE:  {best['mae']:.2f}")
    logger.info(f"  R2:   {best['r2']:.4f}")
    logger.info("ML TRAINING TAMAMLANDI!")
    spark.stop()


if __name__ == '__main__':
    main()
