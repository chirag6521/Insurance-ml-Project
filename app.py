import pandas as pd
import streamlit as st
from src.mlProject.pipeline.prediction import PredictionPipeline

# Create an instance of PredictionPipeline
pipeline = PredictionPipeline()

def main():
    st.title('Insurance Premium Prediction')

    age = st.slider('Enter Age', 18, 100, 18)
    bmi = st.slider('Enter BMI', 15.0, 50.0, 20.0, 0.1)
    children = st.slider('Enter Number of Children', 0, 5, 0)
    sex_male = st.checkbox('Male')
    smoker_yes = st.checkbox('Smoker')
    region_northwest = st.checkbox('Northwest')
    region_southeast = st.checkbox('Southeast')
    region_southwest = st.checkbox('Southwest')

    if st.button('Predict'):
        input_data = {
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'sex_male': [1 if sex_male else 0],
            'smoker_yes': [1 if smoker_yes else 0],
            'region_northwest': [1 if region_northwest else 0],
            'region_southeast': [1 if region_southeast else 0],
            'region_southwest': [1 if region_southwest else 0]
        }
        
        input_df = pd.DataFrame(input_data)
        prediction = pipeline.predict(input_df)  # Call predict method on the pipeline instance
        st.write(f'Predicted Insurance Premium: ${prediction[0]}')

if __name__ == '__main__':
    main()
