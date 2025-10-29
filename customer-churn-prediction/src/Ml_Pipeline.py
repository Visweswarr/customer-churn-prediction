# -*- coding: utf-8 -*- 
"""
Machine Learning Pipeline
Trains and evaluates churn prediction models using Spark MLlib
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, lit, countDistinct, sum, avg, min, max, count, unix_timestamp
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
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

class MLPipeline:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("EcommerceChurnML") \
            .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/user/hive/warehouse") \
            .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
            .enableHiveSupport() \
            .getOrCreate()

        self.spark.sparkContext.setLogLevel("WARN")
        default_fs = self.spark.sparkContext._jsc.hadoopConfiguration().get("fs.defaultFS")
        logger.info("✅ SparkSession initialized successfully. fs.defaultFS = {}".format(default_fs))

        self.feature_cols = [
             "total_events", "view_count", "cart_count", "total_purchase_amount",
             "unique_products_viewed", "unique_categories", "unique_brands",
             "session_duration_minutes", "conversion_rate", "cart_to_purchase_rate",
             "session_efficiency", "high_value_session", "total_sessions",
             "purchase_rate", "avg_spent_per_purchase", "brand_diversity_ratio",
             "long_session", "multi_category_session", "repeat_customer"
        ]
        self.label_col = "is_churned"

    def prepare_data(self):
        """
        Prepare data for ML training from the gold layer table
        """
        logger.info("Preparing data for ML training from gold.ml_churn_dataset")

        try:
            df = self.spark.table("gold.ml_churn_dataset")
            logger.info("Successfully read gold.ml_churn_dataset")
        except Exception as e:
            logger.error("Failed to read gold.ml_churn_dataset: {}".format(e))
            raise

        current_feature_cols = list(self.feature_cols) # Create a mutable list
        logger.info("Initial feature columns: {}".format(current_feature_cols))

        df_clean = df.fillna(0.0, subset=current_feature_cols)
        for col_name in current_feature_cols[:]: # Iterate over a copy
            if col_name in df_clean.columns:
                 # Replace null, NaN, and infinity values with 0.0
                 # For PySpark 3.0 compatibility, check for infinity using comparison
                 df_clean = df_clean.withColumn(col_name,
                    when(col(col_name).isNull() | isnan(col(col_name)) | 
                         (col(col_name) == float('inf')) | (col(col_name) == float('-inf')), 0.0)
                    .otherwise(col(col_name).cast("double")) # Ensure all features are double
                 )
            else:
                 logger.warning("Feature column '{}' not found in DataFrame. Removing from list.".format(col_name))
                 current_feature_cols.remove(col_name)

        self.feature_cols = current_feature_cols # Update the class attribute
        logger.info("Final feature columns used: {}".format(self.feature_cols))

        assembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="features_raw",
            handleInvalid="skip" # Skip rows with nulls if any remain
        )

        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )

        preprocessing_pipeline = Pipeline(stages=[assembler, scaler])
        try:
            logger.info("Fitting preprocessing pipeline...")
            preprocessing_model = preprocessing_pipeline.fit(df_clean)
            df_processed = preprocessing_model.transform(df_clean)
            logger.info("Preprocessing pipeline fitted and data transformed.")
        except Exception as e:
            logger.error("Error during preprocessing pipeline fitting/transforming: {}".format(e))
            df_clean.select(self.feature_cols).summary().show() # Show summary stats for debugging
            raise

        df_ml = df_processed.select("features", self.label_col, "user_id", "user_session").cache()
        logger.info("Cached processed data for ML.")

        train_df, test_df = df_ml.randomSplit([0.8, 0.2], seed=42)
        train_count = train_df.count()
        test_count = test_df.count()
        logger.info("Training set size: {:,}".format(train_count))
        logger.info("Test set size: {:,}".format(test_count))

        df_ml.unpersist()
        logger.info("Unpersisted df_ml.")


        return train_df, test_df, preprocessing_model

    def train_logistic_regression(self, train_df):
        logger.info("Training Logistic Regression model...")
        lr = LogisticRegression(
            featuresCol="features",
            labelCol=self.label_col,
            predictionCol="prediction",
            probabilityCol="probability"
        )
        param_grid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.01, 0.1]) \
            .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
            .build()
        evaluator = BinaryClassificationEvaluator(labelCol=self.label_col, metricName="areaUnderROC")
        cv = CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=2, seed=42)

        cv_model = cv.fit(train_df)
        logger.info("Logistic Regression training completed.")
        return cv_model

    def train_random_forest(self, train_df):
        logger.info("Training Random Forest model...")
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol=self.label_col,
            predictionCol="prediction",
            probabilityCol="probability",
            seed=42,
            numTrees=50
        )
        param_grid = ParamGridBuilder() \
            .addGrid(rf.maxDepth, [5, 10]) \
            .build()
        evaluator = BinaryClassificationEvaluator(labelCol=self.label_col, metricName="areaUnderROC")
        cv = CrossValidator(estimator=rf, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=2, seed=42)

        cv_model = cv.fit(train_df)
        logger.info("Random Forest training completed.")
        return cv_model

    def train_gradient_boosting(self, train_df):
        logger.info("Training Gradient Boosting model...")
        gbt = GBTClassifier(
            featuresCol="features",
            labelCol=self.label_col,
            predictionCol="prediction",
            seed=42,
            maxIter=50
        )
        param_grid = ParamGridBuilder() \
            .addGrid(gbt.maxDepth, [5]) \
            .build()
        evaluator = BinaryClassificationEvaluator(labelCol=self.label_col, metricName="areaUnderROC")
        cv = CrossValidator(estimator=gbt, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=2, seed=42)

        cv_model = cv.fit(train_df)
        logger.info("Gradient Boosting training completed.")
        return cv_model

    def evaluate_model(self, model, test_df, model_name):
        logger.info("Evaluating {} model".format(model_name))
        predictions = model.transform(test_df)

        binary_evaluator_roc = BinaryClassificationEvaluator(labelCol=self.label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        binary_evaluator_pr = BinaryClassificationEvaluator(labelCol=self.label_col, rawPredictionCol="rawPrediction", metricName="areaUnderPR")
        multi_evaluator = MulticlassClassificationEvaluator(labelCol=self.label_col, predictionCol="prediction")

        auc_roc = binary_evaluator_roc.evaluate(predictions)
        auc_pr = binary_evaluator_pr.evaluate(predictions)
        accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
        precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
        recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
        f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})

        logger.info("{} Metrics:".format(model_name))
        logger.info("  AUC-ROC: {:.4f}".format(auc_roc))
        logger.info("  AUC-PR: {:.4f}".format(auc_pr))
        logger.info("  Accuracy: {:.4f}".format(accuracy))
        logger.info("  Precision: {:.4f}".format(precision))
        logger.info("  Recall: {:.4f}".format(recall))
        logger.info("  F1-Score: {:.4f}".format(f1))

        metrics_summary = self.spark.createDataFrame([(model_name, auc_roc, auc_pr, accuracy, precision, recall, f1)],
                                                     ["model_name", "auc_roc", "auc_pr", "accuracy", "precision", "recall", "f1_score"])
        return predictions, metrics_summary

    def get_feature_importance(self, model, model_name, preprocessing_model):
        if hasattr(model.bestModel, 'featureImportances'):
            logger.info("Extracting feature importance for {}".format(model_name))
            assembler = preprocessing_model.stages[0]
            feature_names = assembler.getInputCols()
            importances = model.bestModel.featureImportances.toArray()

            if len(feature_names) != len(importances):
                logger.error("Mismatch between feature names ({}) and importances ({}) for {}".format(len(feature_names), len(importances), model_name))
                return None

            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            logger.info("Top 10 features for {}:".format(model_name))
            for feature, importance in feature_importance[:10]:
                logger.info("  {}: {:.4f}".format(feature, importance))

            importance_df = self.spark.createDataFrame([(model_name, feature, float(importance)) for feature, importance in feature_importance],
                                                       ["model_name", "feature", "importance"])
            return importance_df
        else:
            logger.info("Feature importance not available for {}".format(model_name))
            return None

    def save_results(self, all_metrics, all_predictions_map, feature_importances_list, best_model_name):
        logger.info("Saving model results to Gold Layer")
        logger.info("Saving metrics to gold.model_metrics...")
        (all_metrics.write
            .mode("overwrite")
            .format("parquet")
            .saveAsTable("gold.model_metrics")
        )
        logger.info("✅ Metrics saved.")

        if best_model_name in all_predictions_map:
            logger.info("Saving sampled predictions from best model ({}) to gold.model_predictions...".format(best_model_name))
            best_predictions = all_predictions_map[best_model_name]
            sample_predictions = best_predictions.select("user_id", "user_session", self.label_col, "prediction", "probability").sample(False, 0.1, seed=42)
            (sample_predictions.write
                .mode("overwrite")
                .format("parquet")
                .saveAsTable("gold.model_predictions")
            )
            logger.info("✅ Sampled predictions saved.")
        else:
            logger.warning("Best model '{}' not found in predictions map. Skipping predictions save.".format(best_model_name))

        valid_importances = [df for df in feature_importances_list if df is not None]
        if valid_importances:
            logger.info("Saving feature importances to gold.feature_importance...")
            all_feature_importances = valid_importances[0]
            for i in range(1, len(valid_importances)):
                all_feature_importances = all_feature_importances.union(valid_importances[i])
            (all_feature_importances.write
                .mode("overwrite")
                .format("parquet")
                .saveAsTable("gold.feature_importance")
            )
            logger.info("✅ Feature importances saved.")
        else:
             logger.info("No feature importances to save.")
        logger.info("Model results saving completed.")

    def run_ml_pipeline(self):
        logger.info("--- Starting ML Pipeline Stage ---")
        train_df, test_df, preprocessing_model = self.prepare_data()

        try:
            if self.spark.catalog.isCached("gold.ml_churn_dataset"):
                 self.spark.catalog.uncacheTable("gold.ml_churn_dataset")
                 logger.info("Uncached gold.ml_churn_dataset")
        except Exception as e:
            logger.warning("Could not uncache gold.ml_churn_dataset: {}".format(e))

        models_to_train = {
            "Logistic_Regression": self.train_logistic_regression,
            "Random_Forest": self.train_random_forest,
        }
        trained_models = {}
        all_predictions_map = {}
        metrics_list = []
        feature_importances_list = []

        train_df.cache()
        logger.info("Cached training data.")
        test_df.cache()
        logger.info("Cached test data.")

        for name, train_func in models_to_train.items():
            try:
                model = train_func(train_df)
                trained_models[name] = model
                predictions, metrics = self.evaluate_model(model, test_df, name)
                all_predictions_map[name] = predictions
                metrics_list.append(metrics)
                importance = self.get_feature_importance(model, name, preprocessing_model)
                if importance is not None:
                    feature_importances_list.append(importance)
            except Exception as e:
                logger.error("Failed to train or evaluate {}: {}".format(name, e))
                logger.error(traceback.format_exc())

        train_df.unpersist()
        logger.info("Unpersisted training data.")
        test_df.unpersist()
        logger.info("Unpersisted test data.")

        if not metrics_list:
            logger.error("No models were successfully trained or evaluated.")
            return None, None

        all_metrics = metrics_list[0]
        for i in range(1, len(metrics_list)):
            all_metrics = all_metrics.union(metrics_list[i])

        best_model_row = all_metrics.orderBy(col("auc_roc").desc()).first()
        best_model_name = best_model_row['model_name'] if best_model_row else "N/A"
        auc_value = best_model_row['auc_roc'] if best_model_row else float('nan')
        logger.info("Best model determined: {} with AUC-ROC: {:.4f}".format(best_model_name, auc_value))

        self.save_results(all_metrics, all_predictions_map, feature_importances_list, best_model_name)
        logger.info("--- ML pipeline completed successfully ---")
        return trained_models, all_metrics

    def close(self):
        self.spark.stop()

if __name__ == "__main__":
    pipeline = MLPipeline()
    try:
        trained_models, metrics = pipeline.run_ml_pipeline()
        if metrics:
            logger.info("--- Final Model Comparison ---")
            metrics.show(truncate=False)
        else:
            logger.warning("ML Pipeline did not produce final metrics.")

    except Exception as e:
        logger.error("--- ML Pipeline failed ---")
        logger.error("Error: {}".format(e))
        logger.error(traceback.format_exc())
        raise
    finally:
        pipeline.close()