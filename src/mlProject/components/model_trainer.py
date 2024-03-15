import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from src.mlProject.config.configuration import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        X_train = train_data.drop([self.config.target_column], axis=1)
        y_train = train_data[self.config.target_column]

        models = {
            'LinearRegression': LinearRegression(),
            'ElasticNet': ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42),
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
            'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)

        joblib.dump(models, os.path.join(self.config.root_dir, 'model.joblib'))
