import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessing import load_data, preprocess_data

def train_model(X_train, y_train, preprocessor):
    """
    Train a RandomForest Regressor with GridSearchCV
    """
    # Combine preprocessing and model in a pipeline
    from sklearn.pipeline import Pipeline
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    # Define hyperparameter grid
    # param_grid = {
    #     'regressor__n_estimators': [100, 200, 300],
    #     'regressor__max_depth': [None, 10, 20],
    #     'regressor__min_samples_split': [2, 5, 10]
    # }

    param_grid = {
        'regressor__n_estimators': [100],
        'regressor__max_depth': [10],
        'regressor__min_samples_split': [2]
    }
    
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Performance Metrics:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R² Score: {r2}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def save_model(model, filename='/Users/mahaveer/Desktop/soulai/housing_price_prediction/models/house_price_model.pkl'):
    """
    Save trained model
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# Main execution
if __name__ == '__main__':
    # Load and preprocess data
    df = load_data()
    preprocessed_data = preprocess_data(df)
    
    # Train model
    model = train_model(
        preprocessed_data['X_train'], 
        preprocessed_data['y_train'],
        preprocessed_data['preprocessor']
    )
    
    # Best model
    best_model = model.best_estimator_
    
    # Evaluate model
    eval_results = evaluate_model(
        best_model, 
        preprocessed_data['X_test'], 
        preprocessed_data['y_test']
    )
    print(eval_results)
    # Save model
    save_model(best_model)