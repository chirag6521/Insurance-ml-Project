import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
from src.mlProject.config.configuration import DataTransformationConfig
from src.mlProject.logging import logger

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        # Load the dataset
        data = pd.read_csv(self.config.data_path)

        # Drop any rows with missing values (NaN)
        data.dropna(inplace=True)

        # Convert categorical columns to numerical using one-hot encoding
        categorical_cols = ['sex', 'smoker', 'region']
        data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        # Assuming 'expenses' is the target column
        X = data_encoded.drop('expenses', axis=1)
        y = data_encoded['expenses']

        # Split the data into training and test sets (75% train, 25% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Save the split datasets to CSV files
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        train_data.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test_data.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splitted data into training and test sets")
        logger.info(f"Training data shape: {train_data.shape}")
        logger.info(f"Test data shape: {test_data.shape}")

        print("Training data shape:", train_data.shape)
        print("Test data shape:", test_data.shape)
