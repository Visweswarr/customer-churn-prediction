# -*- coding: utf-8 -*-
"""
Feature Engineering Pipeline - Gold Layer
Creates ML-ready features for churn prediction using a simplified approach
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import logging
import traceback # Import traceback
import time # Import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Basic console handler
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

class FeatureEngineeringPipeline:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("EcommerceFeatureEngineering") \
            .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/user/hive/warehouse") \
            .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
            .enableHiveSupport() \
            .getOrCreate()

        self.spark.sparkContext.setLogLevel("WARN")
        default_fs = self.spark.sparkContext._jsc.hadoopConfiguration().get("fs.defaultFS")
        logger.info("✅ SparkSession initialized successfully. fs.defaultFS = {}".format(default_fs))

    def _hdfs_exists(self, path):
        hconf = self.spark._jsc.hadoopConfiguration()
        fs = self.spark._jvm.org.apache.hadoop.fs.FileSystem.get(hconf)
        p = self.spark._jvm.org.apache.hadoop.fs.Path(path)
        return fs.exists(p)

    def _hdfs_rm(self, path, recursive=True):
        hconf = self.spark._jsc.hadoopConfiguration()
        fs = self.spark._jvm.org.apache.hadoop.fs.FileSystem.get(hconf)
        p = self.spark._jvm.org.apache.hadoop.fs.Path(path)
        if fs.exists(p):
            fs.delete(p, recursive)

    def create_ml_dataset(self, sample_fraction=0.5):
        logger.info("Creating ML-ready dataset with {}% sampling (Gold Layer)".format(sample_fraction*100))
        try:
            df = self.spark.table("silver.ecommerce_cleaned")
            logger.info("Successfully read silver.ecommerce_cleaned")
        except Exception as e:
            logger.error("Failed to read silver.ecommerce_cleaned: {}".format(e))
            raise

        if sample_fraction < 1.0:
            logger.info("Applying {}% sampling...".format(sample_fraction*100))
            df = df.sample(False, sample_fraction, seed=42)
            df.cache()
            logger.info("Sampled {:,} records".format(df.count()))
        else:
             logger.info("Processing full silver dataset (no sampling)")

        logger.info("Aggregating session-level features for ML dataset...")
        ml_features = df.groupBy("user_id", "user_session") \
            .agg(
                count("*").alias("total_events"),
                sum(when(col("event_type") == "view", 1).otherwise(0)).alias("view_count"),
                sum(when(col("event_type") == "cart", 1).otherwise(0)).alias("cart_count"),
                sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("purchase_count"),
                sum(when(col("event_type") == "purchase", col("price")).otherwise(0)).alias("total_purchase_amount"),
                min("event_time").alias("session_start"),
                max("event_time").alias("session_end"),
                countDistinct("product_id").alias("unique_products_viewed"),
                countDistinct("category_code").alias("unique_categories"),
                countDistinct("brand").alias("unique_brands")
            ) \
            .withColumn("is_churned", when(col("purchase_count") == 0, 1).otherwise(0)) \
            .withColumn("session_duration_minutes", (unix_timestamp("session_end") - unix_timestamp("session_start")) / 60.0) \
            .withColumn("conversion_rate", when(col("view_count") > 0, col("purchase_count") / col("view_count")).otherwise(0.0)) \
            .withColumn("cart_to_purchase_rate", when(col("cart_count") > 0, col("purchase_count") / col("cart_count")).otherwise(0.0)) \
            .withColumn("session_efficiency", when(col("total_events") > 0, col("purchase_count") / col("total_events")).otherwise(0.0)) \
            .withColumn("high_value_session", when(col("total_purchase_amount") > 100, 1).otherwise(0)) \
            .fillna(0.0)

        ml_features = ml_features \
            .withColumn("total_sessions", lit(1)) \
            .withColumn("purchase_rate", when(col("total_events") > 0, col("purchase_count") / col("total_events")).otherwise(0.0)) \
            .withColumn("avg_spent_per_purchase", when(col("purchase_count") > 0, col("total_purchase_amount") / col("purchase_count")).otherwise(0.0)) \
            .withColumn("brand_diversity_ratio", when(col("total_events") > 0, col("unique_brands") / col("total_events")).otherwise(0.0)) \
            .withColumn("long_session", when(col("session_duration_minutes") > 60, 1).otherwise(0)) \
            .withColumn("multi_category_session", when(col("unique_categories") > 1, 1).otherwise(0)) \
            .withColumn("repeat_customer", lit(0))

        feature_cols_for_ml = [
             "total_events", "unique_products_viewed", "view_count", "cart_count",
             "session_duration_minutes", "conversion_rate", "cart_to_purchase_rate",
             "unique_brands", "unique_categories", "total_sessions", "purchase_rate",
             "avg_spent_per_purchase", "brand_diversity_ratio", "session_efficiency",
             "high_value_session", "long_session", "multi_category_session", "repeat_customer"
        ]
        ml_features = ml_features.fillna(0.0, subset=feature_cols_for_ml)


        if sample_fraction < 1.0:
             df.unpersist()

        self.spark.sql("CREATE DATABASE IF NOT EXISTS gold")
        logger.info("Saving ML dataset to gold.ml_churn_dataset...")
        self.spark.sql("DROP TABLE IF EXISTS gold.ml_churn_dataset")

        table_schema = """
            user_id INT, user_session STRING, total_events BIGINT, view_count BIGINT, cart_count BIGINT,
            purchase_count BIGINT, total_purchase_amount DOUBLE, session_start TIMESTAMP, session_end TIMESTAMP,
            unique_products_viewed BIGINT, unique_categories BIGINT, unique_brands BIGINT, is_churned INT,
            session_duration_minutes DOUBLE, conversion_rate DOUBLE, cart_to_purchase_rate DOUBLE,
            session_efficiency DOUBLE, high_value_session INT, total_sessions INT, purchase_rate DOUBLE,
            avg_spent_per_purchase DOUBLE, brand_diversity_ratio DOUBLE, long_session INT,
            multi_category_session INT, repeat_customer INT
        """

        hdfs_path = "/user/hive/warehouse/gold.db/ml_churn_dataset"

        logger.info("Writing ML dataset parquet to {}".format(hdfs_path))
        if self._hdfs_exists(hdfs_path):
             logger.info("Removing existing HDFS path: {}".format(hdfs_path))
             self._hdfs_rm(hdfs_path, recursive=True)

        (ml_features.write
            .mode("overwrite")
            .format("parquet")
            .option("compression", "snappy")
            .save(hdfs_path)
        )

        logger.info("Creating EXTERNAL Hive table gold.ml_churn_dataset...")
        self.spark.sql("CREATE EXTERNAL TABLE gold.ml_churn_dataset ({}) STORED AS PARQUET LOCATION '{}'".format(table_schema, hdfs_path))

        self.spark.sql("REFRESH TABLE gold.ml_churn_dataset")

        ml_count = self.spark.table("gold.ml_churn_dataset").count()
        logger.info("✅ ML dataset created in gold.ml_churn_dataset with {:,} records".format(ml_count))
        return self.spark.table("gold.ml_churn_dataset")

    def create_kpi_tables(self):
        logger.info("Creating KPI tables (Gold Layer)")
        try:
            ml_dataset = self.spark.table("gold.ml_churn_dataset")
            logger.info("Using gold.ml_churn_dataset for KPI calculation")
        except Exception as e:
             logger.error("Failed to read gold.ml_churn_dataset for KPIs: {}. Trying silver.ecommerce_cleaned.".format(e))
             try:
                 df_silver = self.spark.table("silver.ecommerce_cleaned")
                 ml_dataset = df_silver.groupBy("user_id", "user_session") \
                    .agg(
                        sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("purchase_count"),
                        sum(when(col("event_type") == "purchase", col("price")).otherwise(0)).alias("total_purchase_amount")
                    ) \
                    .withColumn("is_churned", when(col("purchase_count") == 0, 1).otherwise(0)) \
                    .withColumn("high_value_session", when(col("total_purchase_amount") > 100, 1).otherwise(0)) \
                    .withColumn("repeat_customer", lit(0))
                 logger.warning("KPIs calculated using silver.ecommerce_cleaned (potentially unsampled)")
             except Exception as e_silver:
                 logger.error("Failed to read silver.ecommerce_cleaned as fallback: {}".format(e_silver))
                 return

        churn_by_segments = ml_dataset.groupBy("high_value_session", "repeat_customer") \
            .agg(
                count("*").alias("total_sessions"),
                sum("is_churned").alias("churned_sessions"),
                (sum("is_churned") / count("*")).alias("churn_rate")
            )
        (churn_by_segments.write
            .mode("overwrite")
            .format("parquet")
            .saveAsTable("gold.kpi_churn_by_segments")
        )
        logger.info("✅ Saved KPI: gold.kpi_churn_by_segments")

        overall_kpi = ml_dataset.agg(
                countDistinct("user_id").alias("unique_users_in_sample"),
                count("user_session").alias("total_sessions_in_sample"),
                sum("purchase_count").alias("total_purchases_in_sample"),
                sum("total_purchase_amount").alias("total_revenue_in_sample"),
                avg("is_churned").alias("overall_churn_rate_in_sample")
            )
        (overall_kpi.write
            .mode("overwrite")
            .format("parquet")
            .saveAsTable("gold.kpi_overall_summary_sample")
        )
        logger.info("✅ Saved KPI: gold.kpi_overall_summary_sample")
        logger.info("KPI tables created successfully")

    def validate_features(self):
        logger.info("--- Validating Gold Layer ---")
        tables_to_validate = [
            "gold.ml_churn_dataset",
            "gold.kpi_churn_by_segments",
            "gold.kpi_overall_summary_sample"
        ]
        for table_name in tables_to_validate:
            try:
                time.sleep(1)
                count = self.spark.table(table_name).count()
                logger.info("Table '{}' count: {:,}".format(table_name, count))
                logger.info("Sample data for '{}':".format(table_name))
                self.spark.table(table_name).show(5, truncate=False)
            except Exception as e:
                logger.warning("Could not validate table '{}': {}".format(table_name, e))
        logger.info("--- Validation complete ---")

    def close(self):
        self.spark.stop()

if __name__ == "__main__":
    pipeline = FeatureEngineeringPipeline()
    try:
        logger.info("--- Starting Feature Engineering Stage (Gold Layer) ---")
        SAMPLE_FRACTION_FOR_ML = 1.0 # 100% - Full dataset

        pipeline.create_ml_dataset(sample_fraction=SAMPLE_FRACTION_FOR_ML)
        pipeline.create_kpi_tables()
        pipeline.validate_features()

        logger.info("--- Feature Engineering pipeline completed successfully ---")

    except Exception as e:
        logger.error("--- Feature Engineering Pipeline failed ---")
        logger.error("Error: {}".format(e))
        logger.error(traceback.format_exc())
        raise
    finally:
        pipeline.close()
