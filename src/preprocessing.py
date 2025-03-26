import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

import sys,os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logger import Logger

logger = Logger()

def load_data(file_path=None):
    """
    Load housing dataset, with fallback to California Housing dataset
    """
    if file_path:
        return pd.read_csv(file_path)
    
    logger.info('fetching the california dataset from sklearn')
    housing = fetch_california_housing()
    df = pd.DataFrame(
        data=np.c_[housing.data, housing.target],
        columns=list(housing.feature_names) + ['target']
    )
    return df

def preprocess_data(df):
    """
    Comprehensive data preprocessing pipeline
    """
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    logger.info('Preprocessing the data')
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return {
        'X_train': X_train, 
        'X_test': X_test, 
        'y_train': y_train, 
        'y_test': y_test,
        'preprocessor': preprocessor
    }

def visualize_correlations(df):
    """
    Create correlation heatmap
    """
    logger.info('Mapping the correlation between feature variables and the target variable')
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

# Main execution
if __name__ == '__main__':
    # Load data
    df = load_data()

    
    # Visualize correlations
    visualize_correlations(df)
    
    # Preprocess data
    preprocessed_data = preprocess_data(df)

    print(preprocessed_data)
    
    print("Data Preprocessing Completed!")