# prediction.py
import joblib
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        model_path = Path('artifacts/model_trainer/model.pkl')  # Adjust the path as needed
        print("Loading model from:", model_path)  # Debugging statement
        self.model = joblib.load(model_path)
        print("Model loaded successfully")  # Debugging statement
        print("Type of self.model:", type(self.model))  # Debugging statement

    def predict(self, data):
        # Convert input data to DataFrame if necessary
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest'])

        # Ensure the input data has the correct column order
        data = data[['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']]

        # Convert input data to numpy array
        data_np = data.to_numpy()
        
        # Predict using the model
        prediction = self.model.predict(data_np)

        return prediction
