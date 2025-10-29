# End-to-End Big Data Pipeline: E-commerce Churn Prediction

A complete big data pipeline using Hadoop, Spark, Hive, and MLlib for predicting customer churn in e-commerce.

## 🎯 Project Overview

This project implements a production-ready big data pipeline that processes e-commerce event data to predict customer churn using machine learning. The pipeline follows the medallion architecture (Bronze → Silver → Gold) and achieves 100% accuracy with Random Forest classification.

## 🏗️ Architecture

```
CSV Data → HDFS (Bronze) → Spark Processing (Silver) → Feature Engineering (Gold) → ML Models → Insights
```

### Technology Stack
- **Storage**: Hadoop HDFS, Hive Metastore
- **Processing**: Apache Spark 3.0
- **ML**: Spark MLlib (Logistic Regression, Random Forest)
- **Orchestration**: Docker Compose
- **Analysis**: Jupyter Notebooks, PySpark

## 📊 Data Pipeline Layers

### Dataset
📥 **Download the dataset**: [E-commerce Events Dataset (2019-Oct.csv)](https://drive.google.com/file/d/1xOBEgbzniWURA-ijKmULCDPrjoPA5AEQ/view?usp=sharing)

> Note: The dataset is too large to include in the repository. Download it and place it in the `data/` folder before running the pipeline.

### Bronze Layer (Raw Data)
- **Source**: E-commerce event data (2019-Oct.csv)
- **Format**: Parquet (Snappy compression)
- **Records**: 1,000,000 events
- **Storage**: `hdfs://namenode:9000/user/hive/warehouse/bronze.db/ecommerce_raw`

### Silver Layer (Cleaned & Aggregated)
- **Tables**:
  - `ecommerce_cleaned`: Cleaned events with derived features
  - `product_analytics`: Product-level metrics and conversion rates
- **Transformations**: Data cleaning, type casting, feature extraction

### Gold Layer (ML-Ready & KPIs)
- **Tables**:
  - `ml_churn_dataset`: Session-level features for ML training
  - `model_metrics`: Model performance comparison
  - `feature_importance`: Feature contribution analysis
  - `model_predictions`: Churn predictions
  - `kpi_churn_by_segments`: Churn rates by customer segments
  - `kpi_overall_summary_sample`: Overall business metrics

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- 8GB+ RAM
- 20GB+ disk space

### Setup & Run

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd <project-directory>
```

2. **Start the pipeline**
```cmd
Run_Complete_Pipeline.bat
```

This will:
- Start all Docker containers (Hadoop, Spark, Hive, Jupyter)
- Ingest data into HDFS
- Process through Bronze → Silver → Gold layers
- Train ML models
- Generate KPIs

3. **Access services**
- Jupyter Notebooks: http://localhost:8888
- Spark Master UI: http://localhost:8080
- HDFS NameNode: http://localhost:9870
- YARN ResourceManager: http://localhost:8088

## 📈 Key Results

### Model Performance
| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| **Random Forest** | **1.0000** | **100%** | **1.0000** | **1.0000** | **1.0000** |
| Logistic Regression | 0.9999 | 99.49% | 0.9949 | 0.9949 | 0.9947 |

### Top Feature Importance (Random Forest)
1. **purchase_rate** (39.16%) - Most critical predictor
2. **session_efficiency** (25.72%)
3. **avg_spent_per_purchase** (19.71%)
4. **conversion_rate** (6.79%)
5. **high_value_session** (3.65%)

### Business KPIs
- **Overall Churn Rate**: 95.5%
- **Total Sessions**: 171,578
- **Unique Users**: 130,355
- **Total Revenue**: $2,697,861.72
- **Total Purchases**: 8,367

### Churn by Segments
| Segment | Sessions | Churn Rate |
|---------|----------|------------|
| Low-value, New customers | 165,924 | 98.8% |
| High-value, New customers | 5,654 | 0% |

## 🔍 Project Structure

```
├── src/
│   ├── Data_Ingestion.py          # Bronze layer: CSV → HDFS
│   ├── Data_Processing.py         # Silver layer: Cleaning & aggregation
│   ├── Feature_Engineering.py     # Gold layer: ML features & KPIs
│   └── Ml_Pipeline.py             # Model training & evaluation
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_ML_Results_Visualization.ipynb
│   └── Complete_Analysis_Local.ipynb
├── docker-compose.yml             # Infrastructure setup
├── Dockerfile                     # Spark container config
├── Run_Complete_Pipeline.bat      # Main execution script
└── Check_Status.bat               # Service health check
```

## 💡 Business Insights & Recommendations

### Key Findings
1. **High Churn Rate (95.5%)**: Most users browse without purchasing
2. **Purchase Behavior**: Only 4.9% of sessions result in purchases
3. **High-Value Customers**: Users spending >$100 have 0% churn
4. **Session Efficiency**: Critical metric for predicting conversions

### Immediate Actions (Week 1-2)
1. ✅ Deploy Random Forest model for real-time churn prediction
2. 🎯 Set up alerts for users with >70% churn probability
3. 📧 Create targeted email campaigns for high-risk users
4. 🛒 Implement cart abandonment recovery workflows

### Strategic Initiatives (Month 1-3)
1. **Personalization Engine**: Use feature importance to customize user experience
2. **Dynamic Pricing**: Target high-value session indicators
3. **Engagement Programs**: Increase session efficiency metrics
4. **A/B Testing**: Test interventions on predicted churners

## 🛠️ Technical Features

### Data Processing
- **Sampling**: Configurable (currently 100% of data)
- **Partitioning**: Optimized for Spark processing
- **Compression**: Snappy for efficient storage
- **Schema Evolution**: Supports schema changes

### ML Pipeline
- **Feature Engineering**: 19 engineered features
- **Model Selection**: Cross-validation with 2-fold CV
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Evaluation**: Multiple metrics (AUC-ROC, Precision, Recall, F1)

### Scalability
- Distributed processing with Spark
- HDFS for fault-tolerant storage
- Horizontal scaling via Docker Swarm (optional)
- Handles millions of records efficiently

## 📝 Usage Examples

### View KPIs in Jupyter
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
    .enableHiveSupport() \
    .getOrCreate()

# View churn metrics
spark.sql("SELECT * FROM gold.kpi_churn_by_segments").show()

# View model performance
spark.sql("SELECT * FROM gold.model_metrics").show()

# View top features
spark.sql("""
    SELECT * FROM gold.feature_importance 
    ORDER BY importance DESC 
    LIMIT 10
""").show()
```

### Query with Spark SQL
```bash
docker exec -it spark-master /spark/bin/pyspark \
    --master spark://spark-master:7077 \
    --conf hive.metastore.uris=thrift://hive-metastore:9083
```

## 🧪 Testing & Validation

Run status checks:
```cmd
Check_Status.bat
```

Verify HDFS data:
```bash
docker exec -it namenode hdfs dfs -ls /user/hive/warehouse/gold.db/
```

## 🔧 Configuration

### Adjust Sampling Rate
Edit `src/Feature_Engineering.py`:
```python
SAMPLE_FRACTION_FOR_ML = 1.0  # 100% of data (change to 0.5 for 50%)
```

### Modify ML Models
Edit `src/Ml_Pipeline.py` to add/remove models in `models_to_train` dictionary.

## 📚 Documentation

- **Architecture**: See `docs/architecture.md` (if available)
- **API Reference**: Inline code documentation
- **Notebooks**: Interactive analysis in `notebooks/` folder

## 🤝 Contributing

This is an academic project. For improvements:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This project is for educational purposes.

## 👥 Author

Developed as part of a Big Data Engineering course project.

## 🙏 Acknowledgments

- Dataset: E-commerce behavior data from Kaggle
- Technologies: Apache Hadoop, Spark, Hive ecosystems
- Inspiration: Real-world data engineering practices

## 📞 Support

For issues or questions:
1. Check service status: `Check_Status.bat`
2. View logs: `docker logs <container-name>`
3. Restart services: `docker-compose restart`

---

**Project Status**: ✅ Complete and Production-Ready

**Last Updated**: October 2025
