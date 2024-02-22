import os
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from src.mlProject.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = mean_squared_error(actual, pred, squared=False)
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        
        return rmse, mae, r2

    def save_results(self):
        try:
            test_data = pd.read_csv(self.config.test_data_path)
            model = joblib.load(self.config.model_path)

            test_x = test_data.drop([self.config.target_column], axis=1)
            test_y = test_data[[self.config.target_column]]
            
            predicted_qualities = model.predict(test_x)

            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.config.metric_file_name), exist_ok=True)
            
            # Save metrics as JSON
            metric_results = {"RMSE": rmse, "MAE": mae, "R2": r2}
            with open(self.config.metric_file_name, "w") as metric_file:
                json.dump(metric_results, metric_file)
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")
