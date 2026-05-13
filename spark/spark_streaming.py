"""
============================================================================
Spark Structured Streaming - Kafka to Delta Lake
============================================================================
Bronze/Silver/Gold medallion architecture.
============================================================================
"""

import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, to_timestamp, when, lit,
    count, sum as spark_sum, avg, min as spark_min,
    max as spark_max, window, current_timestamp, trim, upper
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('SparkStreaming')

KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
KAFKA_TOPIC = os.environ.get('KAFKA_TOPIC', 'bank-transactions')
DELTA_LAKE_PATH = os.environ.get('DELTA_LAKE_PATH', '/app/delta-lake')
CHECKPOINT_PATH = f"{DELTA_LAKE_PATH}/checkpoints"
BRONZE_PATH = f"{DELTA_LAKE_PATH}/bronze/bank_transactions"
SILVER_PATH = f"{DELTA_LAKE_PATH}/silver/bank_transactions"
GOLD_PATH = f"{DELTA_LAKE_PATH}/gold/bank_transactions"

KAFKA_MESSAGE_SCHEMA = StructType([
    StructField("message_id", IntegerType(), True),
    StructField("timestamp", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("event_type", StringType(), True),
    StructField("date", StringType(), True),
    StructField("domain", StringType(), True),
    StructField("location", StringType(), True),
    StructField("value", IntegerType(), True),
    StructField("transaction_count", IntegerType(), True)
])


def create_spark_session():
    spark = SparkSession.builder \
        .appName("BankTransactionStreaming") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_PATH) \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark Session created")
    return spark


def read_kafka_stream(spark):
    kafka_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "earliest") \
        .option("maxOffsetsPerTrigger", 10000) \
        .option("failOnDataLoss", "false") \
        .load()

    parsed = kafka_stream.select(
        col("key").cast("string").alias("kafka_key"),
        from_json(col("value").cast("string"), KAFKA_MESSAGE_SCHEMA).alias("data"),
        col("topic"), col("partition"), col("offset"),
        col("timestamp").alias("kafka_timestamp")
    ).select("kafka_key", "data.*", "topic", "partition", "offset", "kafka_timestamp")
    return parsed


def write_bronze_layer(stream_df):
    """Bronze: Raw data as-is"""
    q = stream_df \
        .withColumn("ingestion_timestamp", current_timestamp()) \
        .withColumn("layer", lit("bronze")) \
        .writeStream.format("delta").outputMode("append") \
        .option("checkpointLocation", f"{CHECKPOINT_PATH}/bronze") \
        .option("path", BRONZE_PATH) \
        .trigger(processingTime="10 seconds") \
        .queryName("bronze_writer").start()
    logger.info("Bronze layer streaming started")
    return q


def write_silver_layer(stream_df):
    """Silver: Cleaned and validated data"""
    silver_df = stream_df \
        .filter(col("message_id").isNotNull() & col("domain").isNotNull() &
                col("location").isNotNull() & col("value").isNotNull() &
                col("transaction_count").isNotNull() & col("date").isNotNull()) \
        .withColumn("domain", trim(upper(col("domain")))) \
        .withColumn("location", trim(col("location"))) \
        .withColumn("parsed_date", to_timestamp(col("date"), "yyyy-MM-dd")) \
        .withColumn("event_timestamp", to_timestamp(col("timestamp"))) \
        .filter(col("value") > 0) \
        .filter(col("transaction_count") > 0) \
        .withColumn("ingestion_timestamp", current_timestamp()) \
        .withColumn("layer", lit("silver")) \
        .dropDuplicates(["message_id"])

    q = silver_df.writeStream.format("delta").outputMode("append") \
        .option("checkpointLocation", f"{CHECKPOINT_PATH}/silver") \
        .option("path", SILVER_PATH) \
        .trigger(processingTime="15 seconds") \
        .queryName("silver_writer").start()
    logger.info("Silver layer streaming started")
    return q


def write_gold_layer(stream_df):
    """Gold: Aggregated business metrics"""
    clean_df = stream_df \
        .filter(col("domain").isNotNull() & col("value").isNotNull() & col("date").isNotNull()) \
        .withColumn("parsed_date", to_timestamp(col("date"), "yyyy-MM-dd"))

    gold_df = clean_df \
        .withWatermark("kafka_timestamp", "1 minute") \
        .groupBy(window(col("kafka_timestamp"), "30 seconds"), col("domain"), col("location")) \
        .agg(
            count("*").alias("total_transactions"),
            spark_sum("value").alias("total_value"),
            avg("value").alias("avg_value"),
            spark_min("value").alias("min_value"),
            spark_max("value").alias("max_value"),
            spark_sum("transaction_count").alias("total_transaction_count"),
            avg("transaction_count").alias("avg_transaction_count")
        ) \
        .withColumn("ingestion_timestamp", current_timestamp()) \
        .withColumn("layer", lit("gold"))

    q = gold_df.writeStream.format("delta").outputMode("append") \
        .option("checkpointLocation", f"{CHECKPOINT_PATH}/gold") \
        .option("path", GOLD_PATH) \
        .trigger(processingTime="20 seconds") \
        .queryName("gold_writer").start()
    logger.info("Gold layer streaming started")
    return q


def main():
    logger.info("=" * 60)
    logger.info("SPARK STRUCTURED STREAMING STARTING")
    logger.info("=" * 60)
    spark = create_spark_session()
    stream_df = read_kafka_stream(spark)
    bq = write_bronze_layer(stream_df)
    sq = write_silver_layer(stream_df)
    gq = write_gold_layer(stream_df)
    logger.info("All streaming queries started")
    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        bq.stop(); sq.stop(); gq.stop()
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
