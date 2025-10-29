@echo off
echo ========================================
echo COMPLETE END-TO-END PIPELINE
echo Bronze -> Silver -> Gold -> ML
echo ========================================

echo.
echo Step 1: Starting all Docker containers...
REM Ensure all services are up, especially HDFS, Hive Metastore, and Spark Master/Worker
docker-compose up -d

echo.
echo Step 2: Waiting for services to be ready (Adjust time if needed)...
timeout /t 45 /nobreak

echo.
echo Step 3: Cleaning up previous HDFS data (Idempotent)...
REM Remove previous run data to ensure a clean start. Errors ignored if paths don't exist.
docker exec namenode hdfs dfs -rm -r -f /user/hive/warehouse/bronze.db/ecommerce_raw 2>nul
docker exec namenode hdfs dfs -rm -r -f /user/hive/warehouse/silver.db 2>nul
docker exec namenode hdfs dfs -rm -r -f /user/hive/warehouse/gold.db 2>nul
echo Cleanup complete.

echo.
echo Step 4: Data Ingestion (Bronze Layer)...
echo Loading CSV data into HDFS Parquet and creating Bronze Hive table...
REM Runs Data_Ingestion.py inside the spark-master container
docker exec spark-master /spark/bin/spark-submit ^
  --master local[4] ^
  --driver-memory 6g ^
  --executor-memory 6g ^
  --conf spark.executor.memoryOverhead=2g ^
  --conf spark.sql.shuffle.partitions=20 ^
  /app/Data_Ingestion.py

if %errorlevel% neq 0 (
    echo [ERROR] Data Ingestion failed! Check logs in spark-master container.
    pause
    exit /b 1
)

echo.
echo Step 5: Data Processing (Silver Layer)...
echo Creating cleaned and aggregated Silver Hive tables...
REM Runs Data_Processing.py inside the spark-master container
docker exec spark-master /spark/bin/spark-submit ^
  --master local[4] ^
  --driver-memory 6g ^
  --executor-memory 6g ^
  --conf spark.executor.memoryOverhead=2g ^
  --conf spark.sql.shuffle.partitions=20 ^
  --conf spark.sql.legacy.timeParserPolicy=LEGACY ^
  /app/Data_Processing.py

if %errorlevel% neq 0 (
    echo [ERROR] Data Processing failed! Check logs in spark-master container.
    pause
    exit /b 1
)

echo.
echo Step 6: Feature Engineering (Gold Layer)...
echo Creating ML-ready features and KPI Gold Hive tables...
REM Runs Feature_Engineering.py inside the spark-master container
docker exec spark-master /spark/bin/spark-submit ^
  --master local[4] ^
  --driver-memory 6g ^
  --executor-memory 6g ^
  --conf spark.executor.memoryOverhead=2g ^
  --conf spark.sql.shuffle.partitions=20 ^
  --conf spark.sql.legacy.timeParserPolicy=LEGACY ^
  /app/Feature_Engineering.py

if %errorlevel% neq 0 (
    echo [ERROR] Feature Engineering failed! Check logs in spark-master container.
    pause
    exit /b 1
)

echo.
echo Step 7: ML Pipeline (Model Training)...
echo Training churn prediction models and saving results...
REM Runs Ml_Pipeline.py inside the spark-master container
REM *** ADDED --conf spark.pyspark.python=/usr/bin/python3 ***
docker exec spark-master /spark/bin/spark-submit ^
  --master local[4] ^
  --driver-memory 6g ^
  --executor-memory 6g ^
  --conf spark.executor.memoryOverhead=2g ^
  --conf spark.sql.shuffle.partitions=20 ^
  --conf spark.sql.legacy.timeParserPolicy=LEGACY ^
  --conf spark.pyspark.python=/usr/bin/python3 ^
  /app/Ml_Pipeline.py

if %errorlevel% neq 0 (
    echo [ERROR] ML Pipeline failed! Check logs in spark-master container.
    pause
    exit /b 1
)

echo.
echo Step 8: Verifying final HDFS layers...
echo.
echo Bronze Layer Contents (/user/hive/warehouse/bronze.db/ecommerce_raw):
docker exec namenode hdfs dfs -ls /user/hive/warehouse/bronze.db/ecommerce_raw 2>nul
echo.
echo Silver Layer Contents (/user/hive/warehouse/silver.db/):
docker exec namenode hdfs dfs -ls /user/hive/warehouse/silver.db/ 2>nul
echo.
echo Gold Layer Contents (/user/hive/warehouse/gold.db/):
docker exec namenode hdfs dfs -ls /user/hive/warehouse/gold.db/ 2>nul

echo.
echo ========================================
echo PIPELINE COMPLETE!
echo ========================================
echo.
echo Next Steps:
echo 1. Open http://localhost:8888 (Jupyter Notebook)
echo 2. Explore the created Hive tables in the 'bronze', 'silver', and 'gold' databases using Spark SQL.
echo 3. Analyze ML results (gold.model_metrics, gold.feature_importance, gold.model_predictions).
echo 4. Review KPIs (gold.kpi_...).
echo.
pause