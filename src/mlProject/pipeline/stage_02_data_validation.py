from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.data_validation import DataValidation
from src.mlProject.components.data_validation import DataValidationConfig
from src.mlProject.logging import logger

STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self, config):
        self.config = config

    def main(self):
        data_validation_config = self.config  # Assuming the config object contains data validation configuration
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        config_manager = ConfigurationManager()
        config = config_manager.get_config()  # Assuming you have a method to get the config
        data_validation_pipeline = DataValidationTrainingPipeline(config=config)
        data_validation_pipeline.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e