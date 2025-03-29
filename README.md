# California Housing Price Prediction

This project provides a Flask-based application for predicting California housing prices. It leverages a Machine Learning pipeline, trained on the Scikit-Learn dataset, which includes data preprocessing, model training, and hyperparameter optimization.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Data Analysis](#data-analysis) 
- [Data Preprocessing, Feature Engineering and Pipeline Creation](#data-preprocessing-feature-engineering-and-pipeline-creation)
- [Model Training and Optimization](#model-training-and-optimization)
- [App Deployment and Usage](#app-deployment-and-usage)

## Project Structure
```
├── data/                   # Dataset directory
│   ├── X_train.csv         # Processed training features
│   ├── y_train.csv         # training targets
│   ├── X_test.csv          # Processed test features
│   ├── y_test.csv          # test targets
├── logs/                   # Directory to save logs of runs and model inferences
├── models/                 # Directory for saving trained models
├── notebooks/              # Dataset Analysis
├── pipeline/               # Directory for saving preprocessor pipeline
├── src/
│    ├── preprocessor.py    # Data preprocessing, feature engineering and pipeline creation
│    ├── train.py           # Model training, evaluation, and optimization
├── templates/              # HTML templates for model prediction
├── app.py                  # Flask application
├── logger.py               # Custom Logger
├── requirements.txt        # Dependencies
└── correlation_heatmap     # Correlation heatmap features X target variable
```

## Installation and Setup

- Create and Activate Virtual Environment
```bash
python3.11 -m venv <env_name>
source <env_name>/bin/activate
```

- Install Dependencies
```bash
pip install -r requirements.txt
```
## Data Analysis

-   Median Income shows a strong relationship with the target feature (house price).
-   The EDA notebook (`notebooks/`) provides detailed insights into feature distributions, sample datasets, and overall dataset information.

## Data Preprocessing, Feature Engineering and Pipeline creation
- The dataset is preprocessed and pipeline is generated using `preprocessor.py`.
    - Used custom clustering to group similar locations, then one-hot encoded the clusters. This lets the model better understand location patterns.
    - `rooms_per_bedroom`, `room_per_population`, `population_per_household` are added to contribute more information to the model.
    - Given the limited size of the dataset, simpler modeling approaches were sufficient, and therefore, I didn't use Lasso Regression to remove any features.
    - The numeric columns underwent a three-stage preprocessing procedure. First, missing values were handled using imputation. Second, a log transformation was applied to correct skewed distributions. Finally, the columns were scaled to ensure consistent ranges.
    - The trained data preprocessing pipeline was serialized and saved as `housing_pipeline.joblib` within the pipelines directory to avoid training the pipeline again on training dataset.
    - The processed data is saved as `X_train.csv`, `y_train.csv`, `X_test.csv`, and `y_test.csv` under data folder.


## Model Training and Optimization
- Multiple models are trained using `train.py` initially to filter out which model to select for training, the models are:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
- The best model is selected based on RMSE, MAE, and R2 score.
- Hyperparameter tuning is performed using `RandomizedSearchCV` for optimization for the best model selected.
- The best model fit to data and is saved in `models/best_model.pkl`.
- The log_file under `logs` directory also shows the preocedure of a sample run. 

## App deployment and Usage
- The trained model is deployed using a Flask API (`app.py`).
- To run the  Flask app:
    ```bash
    python3 app.py 
    ```
- After running you should see something like this
  ```bash
    * Serving Flask app 'app'
    * Debug mode: on
    * Running on all addresses (0.0.0.0)
    * Running on http://127.0.0.1:5001
    * Running on http://192.168.1.9:5001
    ........
  ```
- Click on `http://127.0.0.1:5001` to open app in browser
- `/predict` - Accepts the input features in form and returns the predicted price.
- `/predict_json` - Accepts JSON input with housing features and returns predicted price.
