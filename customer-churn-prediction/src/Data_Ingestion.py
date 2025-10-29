# -*- coding: utf-8 -*-
"""
Data Ingestion (Bronze Layer)
- Reads raw CSV from /data (mounted volume)
- Writes Parquet to HDFS under /user/hive/warehouse/bronze.db/ecommerce_raw
- Creates an EXTERNAL Hive table pointing to that HDFS location (idempotent)
"""

import logging
from pyspark.sql import SparkSession, functions as F, types as T
import traceback # Import traceback for detailed error logging
import time # Import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Basic console handler if running standalone might need setup in Spark environment
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

class DataIngestionPipeline:
    def __init__(self):
        self.spark = (
            SparkSession.builder
            .appName("EcommerceDataIngestion")
            .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/user/hive/warehouse")
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
            .config("hive.metastore.uris", "thrift://hive-metastore:9083")
            .enableHiveSupport()
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("✅ SparkSession initialized successfully.")

        self.db_name = "bronze"
        self.table_name = "ecommerce_raw"
        self.db_path = "hdfs://namenode:9000/user/hive/warehouse/bronze.db"
        self.table_path = "{}/{}".format(self.db_path, self.table_name)

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

    def ingest_raw_data(self, input_path, hdfs_output_path=None, overwrite=True, sample_rows=None):
        """
        Read raw CSV, sample if requested, and write to HDFS as Parquet.
        """
        output = hdfs_output_path or self.table_path

        logger.info("Starting data ingestion from {}".format(input_path))
        reader = (
            self.spark.read
            .option("header", True)
            .option("mode", "DROPMALFORMED")
            .option("inferSchema", True)
        )

        df_raw = reader.csv(input_path)

        if sample_rows:
            logger.info("Sampling {:,} rows...".format(sample_rows))
            df_raw = df_raw.limit(int(sample_rows))
            df_raw.cache()
            logger.info("Sampled dataframe count: {:,}".format(df_raw.count()))

        schema = T.StructType([
            T.StructField("event_time",   T.StringType(),  True),
            T.StructField("event_type",   T.StringType(),  True),
            T.StructField("product_id",   T.IntegerType(), True),
            T.StructField("category_id",  T.LongType(),    True),
            T.StructField("category_code",T.StringType(),  True),
            T.StructField("brand",        T.StringType(),  True),
            T.StructField("price",        T.DoubleType(),  True),
            T.StructField("user_id",      T.IntegerType(), True),
            T.StructField("user_session", T.StringType(),  True),
        ])

        df = df_raw.select(
                F.col("event_time").cast("string"),
                F.col("event_type").cast("string"),
                F.col("product_id").cast("int"),
                F.col("category_id").cast("long"),
                F.col("category_code").cast("string"),
                F.col("brand").cast("string"),
                F.col("price").cast("double"),
                F.col("user_id").cast("int"),
                F.col("user_session").cast("string"),
            )

        df = df.na.fill({'price': 0.0})
        df = df.filter(F.col("user_id").isNotNull() & F.col("product_id").isNotNull())

        logger.info("Final Schema before writing:")
        df.printSchema()

        if overwrite and self._hdfs_exists(output):
            logger.info("Output path exists, removing for overwrite: {}".format(output))
            self._hdfs_rm(output, recursive=True)

        (
            df.write
            .mode("overwrite" if overwrite else "error")
            .option("compression", "snappy")
            .parquet(output)
        )
        logger.info("✅ Wrote bronze parquet to {}".format(output))
        if sample_rows:
            df_raw.unpersist()

        self.spark.catalog.refreshByPath(output)

    def create_hive_external_table(self, location=None):
        loc = location or self.table_path
        self.spark.sql("CREATE DATABASE IF NOT EXISTS {} LOCATION '{}'".format(self.db_name, self.db_path))
        self.spark.sql("DROP TABLE IF EXISTS {}.{}".format(self.db_name, self.table_name))
        self.spark.sql("""
            CREATE EXTERNAL TABLE {}.{} (
                event_time    STRING,
                event_type    STRING,
                product_id    INT,
                category_id   BIGINT,
                category_code STRING,
                brand         STRING,
                price         DOUBLE,
                user_id       INT,
                user_session  STRING
            )
            STORED AS PARQUET
            LOCATION '{}'
        """.format(self.db_name, self.table_name, loc))

        self.spark.sql("REFRESH TABLE {}.{}".format(self.db_name, self.table_name))

        logger.info("✅ External table {}.{} is ready at {}".format(self.db_name, self.table_name, loc))

    def validate_ingestion(self):
        try:
            time.sleep(2) # Give metastore a moment
            cnt = self.spark.sql("SELECT COUNT(*) AS c FROM {}.{}".format(self.db_name, self.table_name)).collect()[0]["c"]
            logger.info("✅ Validation: {}.{} has {:,} rows".format(self.db_name, self.table_name, cnt))
            logger.info("Sample data:")
            self.spark.sql("SELECT * FROM {}.{} LIMIT 5".format(self.db_name, self.table_name)).show()
        except Exception as e:
            logger.error("❌ Validation failed for {}.{}: {}".format(self.db_name, self.table_name, e))
            try:
                self.spark.sql("REFRESH TABLE {}.{}".format(self.db_name, self.table_name))
                logger.info("Refreshed table {}.{} again.".format(self.db_name, self.table_name))
                cnt = self.spark.sql("SELECT COUNT(*) AS c FROM {}.{}".format(self.db_name, self.table_name)).collect()[0]["c"]
                logger.info("✅ Validation (after refresh): {}.{} has {:,} rows".format(self.db_name, self.table_name, cnt))
            except Exception as e_retry:
                 logger.error("❌ Validation failed even after retry: {}".format(e_retry))
                 raise e 

    def close(self):
        try:
            self.spark.stop()
            logger.info("🧹 Spark session closed successfully.")
        except Exception:
            pass

if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    try:
        SAMPLE_SIZE = 1000000 # 1 million rows (NO underscores)
        # SAMPLE_SIZE = None
        INPUT_CSV_PATH = "file:///data/Ecommerce_Csv_2019-Oct.csv"

        log_level = "sampled ({:,} rows)".format(SAMPLE_SIZE) if SAMPLE_SIZE else "full dataset"
        logger.info("--- Starting Ingestion Stage ({}) ---".format(log_level))

        pipeline.ingest_raw_data(
            INPUT_CSV_PATH,
            sample_rows=SAMPLE_SIZE,
            overwrite=True
        )
        pipeline.create_hive_external_table()
        pipeline.validate_ingestion()

        logger.info("--- Data Ingestion pipeline completed successfully ---")

    except Exception as e:
        logger.error("--- Pipeline failed ---")
        logger.error("Error: {}".format(e))
        logger.error(traceback.format_exc())
        raise
    finally:
        pipeline.close()
