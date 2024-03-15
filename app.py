import streamlit as st
import pandas as pd
from src.mlProject.pipeline.prediction import predict_expenses

def main():
    
    st.title('Insurance Expenses Prediction')

    st.sidebar.header('User Input')

    age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=30)
    bmi = st.sidebar.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
    children = st.sidebar.number_input('Number of Children', min_value=0, max_value=10, value=0)
    sex_male = st.sidebar.radio('Sex (Male)', [0, 1], index=1)  # 1 for male, 0 for female
    smoker_yes = st.sidebar.radio('Smoker (Yes)', [0, 1], index=0)  # 1 for smoker, 0 for non-smoker
    region_northwest = st.sidebar.radio('Region (Northwest)', [0, 1], index=0)  # One-hot encoded region features
    region_southeast = st.sidebar.radio('Region (Southeast)', [0, 1], index=0)
    region_southwest = st.sidebar.radio('Region (Southwest)', [0, 1], index=0)

    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex_male': [sex_male],
        'smoker_yes': [smoker_yes],
        'region_northwest': [region_northwest],
        'region_southeast': [region_southeast],
        'region_southwest': [region_southwest]
    })

    if st.sidebar.button('Predict'):
        predicted_expenses = predict_expenses(input_data)
        
        st.write('Predicted Expenses:', predicted_expenses)

if __name__ == '__main__':
    main()