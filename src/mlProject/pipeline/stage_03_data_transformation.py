import os
import pandas as pd
from src.mlProject.entity.config_entity import DataTransformationConfig
from src.mlProject.logging import logger
from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.data_transformation import DataTransformation

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.encoder = None

    def main(self):
        categorical_features = self.config.categorical_features  # Accessing categorical_features from the config
        data_transformation = DataTransformation(config=self.config)
        data_transformation.train_test_spliting(categorical_features)

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    config_manager = ConfigurationManager()
    config = config_manager.get_data_transformation_config()  # Get the data transformation config
    data_transformation_pipeline = DataTransformationTrainingPipeline(config)
    data_transformation_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
