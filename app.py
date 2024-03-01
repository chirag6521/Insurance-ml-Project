from flask import Flask, request, jsonify
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
encoder = joblib.load(Path('artifacts/data_transformation/encoder.pkl'))

def preprocess_input(age, bmi, children, sex, smoker, region):
    # Encode categorical features
    sex_encoded = 1 if sex == 'male' else 0
    smoker_encoded = 1 if smoker == 'yes' else 0
    region_encoded = 1 if region == 'southwest' else 0  # Assuming 'southwest' as an example

    # Create input array with 11 features
    data = np.array([[age, bmi, children, 0, 0, 0, sex_encoded, smoker_encoded, region_encoded, 0, 0]])

    # Perform one-hot encoding
    encoded_data = encoder.transform(data).toarray()

    return encoded_data

@app.route('/predict', methods=['POST'])
def predict():
    # Get request data
    data = request.get_json()
    
    # Extract input values
    age = data['age']
    bmi = data['bmi']
    children = data['children']
    sex = data['sex']
    smoker = data['smoker']
    region = data['region']
    
    # Preprocess input
    processed_data = preprocess_input(age, bmi, children, sex, smoker, region)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    # Print prediction and input values for debugging
    print('Input values:', data)
    print('Prediction:', prediction[0])
    
    # Return prediction
    response = {'prediction': prediction[0]}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)