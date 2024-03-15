from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import pandas as pd
import joblib
import json
from src.mlProject.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = mean_squared_error(actual, pred, squared=False)
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        accuracy = r2 * 100
        return rmse, mae, r2, accuracy

    def save_results(self):
        try:
            test_data = pd.read_csv(self.config.test_data_path)
            models = joblib.load(self.config.model_path)
            metrics = {}

            for name, model in models.items():
                if hasattr(model, 'predict'):
                    test_x = test_data.drop([self.config.target_column], axis=1)
                    test_y = test_data[self.config.target_column]
                    predicted_qualities = model.predict(test_x)

                    rmse, mae, r2, accuracy = self.eval_metrics(test_y, predicted_qualities)

                    metrics[name] = {"rmse": rmse}
                    metrics[name]["mae"] = mae
                    metrics[name]["r2"] = r2
                    metrics[name]["accuracy"] = accuracy

                else:
                    print(f"Error: Model {name} does not have a 'predict' method.")

            with open(self.config.metric_file_name, 'w') as file:
                json.dump(metrics, file, indent=4)

        except Exception as e:
            print(f"Error occurred while saving results: {str(e)}")
