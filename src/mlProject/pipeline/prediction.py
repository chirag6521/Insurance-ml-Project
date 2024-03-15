import pandas as pd
import joblib

model = joblib.load('artifacts/model_trainer/model.joblib')

def predict_expenses(input_data):
    
    predicted_expenses = model.predict(input_data)
    return predicted_expenses