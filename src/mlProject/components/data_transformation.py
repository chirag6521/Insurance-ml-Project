import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib
from pathlib import Path

class DataTransformationConfig:
    def __init__(self, root_dir: str, data_path: str, categorical_features: list):
        self.root_dir = root_dir
        self.data_path = data_path
        self.categorical_features = categorical_features

class DataTransformation:
    def __init__(self, config):
        self.config = config
        
    def train_test_spliting(self, categorical_features):
        data_path = self.config.data_path
        
        # Load the data
        data = pd.read_csv(data_path)

        # Separate features and target variable
        X = data.drop(columns=['expenses'])
        y = data['expenses']

        # One-hot encode categorical features
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_features]))
        X_encoded.columns = encoder.get_feature_names_out(input_features=categorical_features)
        X.drop(columns=categorical_features, inplace=True)
        X = pd.concat([X, X_encoded], axis=1)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Add 'expenses' column back to train and test data
        X_train['expenses'] = y_train
        X_test['expenses'] = y_test

        # Create the directory if it doesn't exist
        data_transformation_dir = Path(self.config.root_dir) / 'artifacts' / 'data_transformation'
        data_transformation_dir.mkdir(parents=True, exist_ok=True)

        # Save the encoder inside data_transformation directory
        encoder_path = data_transformation_dir / 'encoder.pkl'
        joblib.dump(encoder, encoder_path)

        # Save the transformed training and test data inside data_transformation directory
        train_csv_path = data_transformation_dir / 'train.csv'
        test_csv_path = data_transformation_dir / 'test.csv'
        X_train.to_csv(train_csv_path, index=False)
        X_test.to_csv(test_csv_path, index=False)
