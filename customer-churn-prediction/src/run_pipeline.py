# -*- coding: utf-8 -*-
"""
Complete Big Data Pipeline Orchestration (Python Version)
Runs all stages sequentially using subprocess.
NOTE: This is generally NOT used with the docker-compose setup.
Use Run_Complete_Pipeline.bat instead for Docker execution via spark-submit.
This script might be intended for running directly on a machine with Python & Spark configured.
"""

import subprocess
import sys
import logging
from datetime import datetime
import traceback # Import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Basic console handler
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

def run_stage(stage_name, script_path):
    """Run a pipeline stage as a separate Python process and handle errors"""
    logger.info("{}".format('='*60)) # Corrected f-string
    logger.info("Starting Stage: {}".format(stage_name)) # Corrected f-string
    logger.info("{}".format('='*60)) # Corrected f-string

    start_time = datetime.now()

    try:
        result = subprocess.run(
            [sys.executable, script_path], 
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info("--- Stage Output for {} ---".format(stage_name))
        logger.info(result.stdout)
        if result.stderr:
            logger.warning("--- Stage Stderr for {} ---".format(stage_name))
            logger.warning(result.stderr)
        logger.info("✅ {} completed successfully in {:.2f}s".format(stage_name, duration)) # Corrected f-string
        return True

    except subprocess.CalledProcessError as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error("❌ {} failed after {:.2f}s".format(stage_name, duration)) # Corrected f-string
        logger.error("--- Stage Error Output ---")
        logger.error(e.stderr) 
        return False
    except Exception as e_general:
         duration = (datetime.now() - start_time).total_seconds()
         logger.error("❌ An unexpected error occurred in {} after {:.2f}s".format(stage_name, duration)) # Corrected f-string
         logger.error("Error details: {}".format(e_general)) # Corrected f-string
         logger.error(traceback.format_exc())
         return False


def main():
    """Run complete pipeline using Python subprocess"""
    logger.info("="*60)
    logger.info("PYTHON PIPELINE ORCHESTRATOR")
    logger.info("(Intended for non-Docker or specific execution environments)")
    logger.info("="*60)

    pipeline_start = datetime.now()

    script_dir = "/app" 
    stages = [
        ("Data Ingestion (Bronze Layer)", "{}/Data_Ingestion.py".format(script_dir)), # Corrected f-string
        ("Data Processing (Silver Layer)", "{}/Data_Processing.py".format(script_dir)), # Corrected f-string
        ("Feature Engineering (Gold Layer)", "{}/Feature_Engineering.py".format(script_dir)), # Corrected f-string
        ("ML Pipeline (Model Training)", "{}/Ml_Pipeline.py".format(script_dir)) # Corrected f-string
    ]

    for stage_name, script_path in stages:
        if not run_stage(stage_name, script_path):
            logger.error("Pipeline failed at stage: {}".format(stage_name)) # Corrected f-string
            sys.exit(1)

    total_duration = (datetime.now() - pipeline_start).total_seconds()

    logger.info("="*60)
    logger.info("✅ PYTHON ORCHESTRATED PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("Total Duration: {:.2f} minutes".format(total_duration/60.0)) # Corrected f-string
    logger.info("="*60)

if __name__ == "__main__":
    main()
