from src.mlProject.logging import logger
from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.mlProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.mlProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.mlProject.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from src.mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
    config = {...} 
    data_ingestion = DataIngestionTrainingPipeline(config=config)
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation stage"
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        config_manager = ConfigurationManager()
        config = config_manager.get_data_validation_config()
        data_validation_pipeline = DataValidationTrainingPipeline(config=config)
        data_validation_pipeline.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
    
STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   config_manager = ConfigurationManager()
   config = config_manager.get_data_transformation_config()
   data_ingestion = DataTransformationTrainingPipeline(config=config)
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Trainer stage"
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        config_manager = ConfigurationManager()
        config = config_manager.get_model_trainer_config()
        model_trainer = ModelEvaluationTrainingPipeline(config=config)
        model_trainer.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == '__main__':
    try:
        config_manager = ConfigurationManager()
        config = config_manager.get_model_evaluation_config()
        model_evaluation = ModelEvaluationTrainingPipeline(config=config)
        model_evaluation.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e