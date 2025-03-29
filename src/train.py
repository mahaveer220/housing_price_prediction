import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

import os,sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logger import Logger
logger = Logger()

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
DATA_DIR = os.path.join(ROOT_DIR,"data")
MODEL_DIR = os.path.join(ROOT_DIR,"models")

def load_data():
    data_dir = DATA_DIR
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()
    return X_train, y_train, X_test, y_test

class ModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor(),
        }
        self.best_model = None
        self.best_model_name = ""
        self.best_params = None

    def evaluate_model(self, model):
        y_pred = model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return rmse, mae, r2

    def train_and_evaluate(self):
        results = {}
        logger.info("Training and evaluating models...")
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            results[name] = self.evaluate_model(model)
            
        logger.info("Model evaluation results:")
        for name, metrics in results.items():
            rmse, mae, r2 = metrics
            logger.info(f"{name}: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        best_model_name = min(results, key=lambda k: results[k][0]) 
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        logger.info(f"Best Model: {best_model_name} with RMSE: {results[best_model_name][0]:.4f}, MAE: {results[best_model_name][1]:.4f}, R2: {results[best_model_name][2]:.4f}")
        return results

    def evaluate_on_test_set(self):
        logger.info(f"Evaluating {self.best_model_name} on test set...")
        y_pred = self.best_model.predict(self.X_test)
        
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        logger.info(f"Test Set Evaluation - {self.best_model_name}:")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"R2 Score: {r2:.4f}")
        
        return {"RMSE": rmse, "MAE": mae, "R2": r2}

    def optimize_model(self):
        # param_grid = {
        # "Random Forest": {
        # # "n_estimators": [50, 100, 200],  # Number of trees
        # # "max_depth": [None, 10, 20, 30],  # Tree depth limit
        # # "min_samples_split": [2, 5, 10],  # Min samples to split a node
        # # "min_samples_leaf": [1, 2, 4],  # Min samples in a leaf
        # # "max_features": ["sqrt", "log2"],  # Number of features to consider per split
        # # "criterion": ["gini", "entropy"],  # Splitting criterion
        # # "random_state": [42]  # Ensure reproducibility
        # }}
        param_grid = {
           "XGBoost": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": np.linspace(0.01, 0.3, 10),  # Continuous range
            "subsample": np.linspace(0.5, 1, 5),  # Values between 0.5 and 1
            "colsample_bytree": np.linspace(0.5, 1, 5),
        }}

        
        if self.best_model_name in param_grid:
            # grid_search = GridSearchCV(self.best_model, param_grid[self.best_model_name], scoring='neg_mean_squared_error', cv=4)
            # grid_search.fit(self.X_train, self.y_train)
            # self.best_model = grid_search.best_estimator_
            # self.best_params = grid_search.best_params_
            # logger.info(f"Optimized {self.best_model_name} with Params: {self.best_params}")

            random_search = RandomizedSearchCV(
            self.best_model,
            param_distributions=param_grid[self.best_model_name],
            n_iter=20,
            scoring='r2',
            cv=5,
            verbose=2,
            n_jobs=-1,
            random_state=42
            )

            random_search.fit(self.X_train, self.y_train)
            self.best_model = random_search.best_estimator_
            self.best_params = random_search.best_params_
            logger.info(f"Optimized {self.best_model_name} with Params: {self.best_params}")
            

    def save_best_model(self, path=os.path.join(MODEL_DIR, "best_model.pkl")):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        with open(path, "wb") as f:
            pickle.dump(self.best_model, f)
        logger.info(f"Best model saved at {path}")
        logger.info(f"Best Model Parameters: {self.best_params}")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    trainer = ModelTrainer(X_train, y_train, X_test, y_test)
    trainer.train_and_evaluate()
    trainer.optimize_model()
    trainer.save_best_model()
    trainer.evaluate_on_test_set()



