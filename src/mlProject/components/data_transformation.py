import os
from src.mlProject.logging import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from src.mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.encoder = None

    def train_test_split_and_encode(self):
        data = pd.read_csv(self.config.data_path)

        # Define categorical columns
        categorical_cols = ['sex', 'smoker', 'region']
        target_column = 'expenses'

        # Encode categorical columns
        data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        # Ensure 'expenses' column is included in the encoded data
        if target_column not in data_encoded.columns:
            raise ValueError(f"'{target_column}' column is missing in the encoded data")

        # Split the data into features (X) and target variable (y)
        X = data_encoded.drop(columns=[target_column])
        y = data_encoded[target_column]

        # Check if the lengths of X and y match
        if len(X) != len(y):
            raise ValueError("Length of features (X) does not match length of target variable (y)")

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Save the encoder
        encoder_path = os.path.join(self.config.root_dir, "encoder.pkl")
        joblib.dump(self.encoder, encoder_path)

        # Save the transformed data as CSV files
        train_csv_path = os.path.join(self.config.root_dir, "train.csv")
        test_csv_path = os.path.join(self.config.root_dir, "test.csv")
        pd.concat([X_train, y_train], axis=1).to_csv(train_csv_path, index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(test_csv_path, index=False)

        return train_csv_path, test_csv_path, encoder_path
