# -*- coding: utf-8 -*-
"""
Data Processing Pipeline - Silver Layer
Cleans, transforms, and enriches the raw data
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
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

class DataProcessingPipeline:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("EcommerceDataProcessing") \
            .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/user/hive/warehouse") \
            .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.executor.memory", "6g") \
            .config("spark.driver.memory", "6g") \
            .config("spark.executor.memoryOverhead", "2g") \
            .config("spark.memory.fraction", "0.8") \
            .config("spark.memory.storageFraction", "0.3") \
            .config("spark.sql.shuffle.partitions", "20") \
            .config("spark.default.parallelism", "20") \
            .config("spark.sql.files.maxPartitionBytes", "134217728") \
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

    def clean_raw_data(self):
        logger.info("Starting data cleaning process (Silver Layer)")
        try:
            df = self.spark.table("bronze.ecommerce_raw")
            logger.info("Successfully read from bronze.ecommerce_raw")
        except Exception as e:
            logger.error("Failed to read from bronze.ecommerce_raw: {}".format(e))
            raise

        df_cleaned = df \
            .withColumn("event_time", to_timestamp(col("event_time"), "yyyy-MM-dd HH:mm:ss 'UTC'")) \
            .filter(col("event_time").isNotNull()) \
            .fillna({
                "category_code": "unknown_category",
                "brand": "unknown_brand",
                "user_session": "unknown_session",
                "price": 0.0
            }) \
            .filter(col("user_id").isNotNull()) \
            .filter(col("product_id").isNotNull()) \
            .filter(col("event_type").isin(["view", "cart", "purchase"])) \
            .filter(col("price") >= 0)

        df_enriched = df_cleaned \
            .withColumn("event_date", to_date("event_time")) \
            .withColumn("event_year", year("event_time")) \
            .withColumn("event_month", month("event_time")) \
            .withColumn("event_day", dayofmonth("event_time")) \
            .withColumn("event_hour", hour("event_time")) \
            .withColumn("event_weekday", dayofweek("event_time")) \
            .withColumn("price_range",
                when(col("price") <= 0, "free")
                .when(col("price") < 10, "0-10")
                .when(col("price") < 50, "10-50")
                .when(col("price") < 200, "50-200")
                .otherwise("200+")) \
            .withColumn("processing_timestamp", current_timestamp())

        self.spark.sql("CREATE DATABASE IF NOT EXISTS silver")
        logger.info("Writing cleaned data to silver.ecommerce_cleaned...")
        
        (df_enriched.write
            .mode("overwrite")
            .format("parquet")
            .option("compression", "snappy")
            .partitionBy("event_date")
            .saveAsTable("silver.ecommerce_cleaned")
        )

        record_count = self.spark.table("silver.ecommerce_cleaned").count()
        logger.info("✅ Cleaned data saved to silver.ecommerce_cleaned. Record count: {:,}".format(record_count))
        return self.spark.table("silver.ecommerce_cleaned")

    def create_user_sessions(self):
        logger.warning("Starting user session aggregations (can be memory intensive)")
        try:
            df = self.spark.table("silver.ecommerce_cleaned")
            logger.info("Loaded silver.ecommerce_cleaned table for session aggregation")
            df.cache()
            logger.info("Cached dataframe with {:,} records".format(df.count()))

            logger.info("Starting session aggregation...")
            session_agg = df.groupBy("user_id", "user_session") \
                .agg(
                    count("*").alias("total_events"),
                    countDistinct("product_id").alias("unique_products_viewed"),
                    sum(when(col("event_type") == "view", 1).otherwise(0)).alias("view_count"),
                    sum(when(col("event_type") == "cart", 1).otherwise(0)).alias("cart_count"),
                    sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("purchase_count"),
                    sum(when(col("event_type") == "purchase", col("price")).otherwise(0)).alias("total_purchase_amount"),
                    avg(when(col("event_type") == "purchase", col("price"))).alias("avg_purchase_price"),
                    min("event_time").alias("session_start"),
                    max("event_time").alias("session_end"),
                    countDistinct("brand").alias("unique_brands"),
                    countDistinct("category_code").alias("unique_categories")
                )

            logger.info("Adding computed columns to session aggregation...")
            session_agg = session_agg \
                .withColumn("session_duration_minutes", (unix_timestamp("session_end") - unix_timestamp("session_start")) / 60.0) \
                .withColumn("conversion_rate", when(col("view_count") > 0, col("purchase_count") / col("view_count")).otherwise(0.0)) \
                .withColumn("cart_to_purchase_rate", when(col("cart_count") > 0, col("purchase_count") / col("cart_count")).otherwise(0.0)) \
                .fillna(0.0)

            logger.info("Writing user sessions to silver.user_sessions...")
            (session_agg.write
                .mode("overwrite")
                .format("parquet")
                .option("compression", "snappy")
                .saveAsTable("silver.user_sessions")
            )

            df.unpersist()
            session_count = self.spark.table("silver.user_sessions").count()
            logger.info("✅ User session table created successfully. Session count: {:,}".format(session_count))
            return self.spark.table("silver.user_sessions")

        except Exception as e:
            logger.error("Failed to create user sessions: {}".format(e))
            logger.error(traceback.format_exc())
            try: df.unpersist()
            except: pass
            raise

    def create_product_analytics(self):
        logger.info("Creating product analytics (Silver Layer)")
        try:
            df = self.spark.table("silver.ecommerce_cleaned")
            logger.info("Starting product aggregation...")
            product_agg = df.groupBy("product_id", "category_code", "brand") \
                .agg(
                    count("*").alias("total_interactions"),
                    countDistinct("user_id").alias("unique_users"),
                    sum(when(col("event_type") == "view", 1).otherwise(0)).alias("view_count"),
                    sum(when(col("event_type") == "cart", 1).otherwise(0)).alias("cart_count"),
                    sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("purchase_count"),
                    sum(when(col("event_type") == "purchase", col("price")).otherwise(0)).alias("total_revenue"),
                    avg(when(col("event_type") == "purchase", col("price"))).alias("avg_price"),
                    max("price").alias("max_price"),
                    min(when(col("price") > 0, col("price"))).alias("min_selling_price")
                ) \
                .withColumn("view_to_cart_rate", when(col("view_count") > 0, col("cart_count") / col("view_count")).otherwise(0.0)) \
                .withColumn("cart_to_purchase_rate", when(col("cart_count") > 0, col("purchase_count") / col("cart_count")).otherwise(0.0)) \
                .withColumn("overall_conversion_rate", when(col("view_count") > 0, col("purchase_count") / col("view_count")).otherwise(0.0)) \
                .fillna(0.0)

            db_name = "silver"
            table_name = "product_analytics"
            hdfs_path = "hdfs://namenode:9000/user/hive/warehouse/{}.db/{}".format(db_name, table_name)
            logger.info("Writing product analytics parquet to {}".format(hdfs_path))

            if self._hdfs_exists(hdfs_path):
                logger.info("Removing existing HDFS path: {}".format(hdfs_path))
                self._hdfs_rm(hdfs_path, recursive=True)

            (product_agg.write
                .mode("overwrite")
                .format("parquet")
                .option("compression", "snappy")
                .save(hdfs_path)
            )

            logger.info("Creating EXTERNAL Hive table silver.product_analytics...")
            self.spark.sql("DROP TABLE IF EXISTS {}.{}".format(db_name, table_name))

            table_schema = """
                product_id INT, category_code STRING, brand STRING, total_interactions BIGINT,
                unique_users BIGINT, view_count BIGINT, cart_count BIGINT, purchase_count BIGINT,
                total_revenue DOUBLE, avg_price DOUBLE, max_price DOUBLE, min_selling_price DOUBLE,
                view_to_cart_rate DOUBLE, cart_to_purchase_rate DOUBLE, overall_conversion_rate DOUBLE
            """
            self.spark.sql("""
                CREATE EXTERNAL TABLE {}.{} ({})
                STORED AS PARQUET
                LOCATION '{}'
            """.format(db_name, table_name, table_schema, hdfs_path))

            self.spark.sql("REFRESH TABLE {}.{}".format(db_name, table_name))

            product_count = self.spark.table("silver.product_analytics").count()
            logger.info("✅ Product analytics created successfully. Product count: {:,}".format(product_count))
            return self.spark.table("silver.product_analytics")

        except Exception as e:
            logger.error("Failed to create product analytics: {}".format(e))
            logger.error(traceback.format_exc())
            raise

    def validate_processing(self):
        logger.info("--- Validating Silver Layer ---")
        tables_to_validate = ["silver.ecommerce_cleaned", "silver.user_sessions", "silver.product_analytics"]
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
    pipeline = DataProcessingPipeline()
    try:
        logger.info("--- Starting Processing Stage (Silver Layer) ---")
        pipeline.clean_raw_data()
        # logger.info("Attempting user_sessions aggregation (may fail on large data)...")
        # pipeline.create_user_sessions()
        logger.warning("Skipping user_sessions aggregation due to potential size/memory issues. Uncomment in script if needed.")
        pipeline.create_product_analytics()
        pipeline.validate_processing()
        logger.info("--- Data Processing pipeline completed successfully ---")

    except Exception as e:
        logger.error("--- Processing Pipeline failed ---")
        logger.error("Error: {}".format(e))
        logger.error(traceback.format_exc())
        raise
    finally:
        pipeline.close()
