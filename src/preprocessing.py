import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder, FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from sklearn.datasets import fetch_california_housing
import os,sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logger import Logger

logger = Logger()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(BASE_DIR,'../pipeline')
DATA_DIR = os.path.join(BASE_DIR,'../data')

class ClusterAdder(BaseEstimator, TransformerMixin):
        def __init__(self, n_clusters=4, lat_col='Latitude', lon_col='Longitude', cluster_col='location_cluster', random_state=42):
            self.n_clusters = n_clusters
            self.lat_col = lat_col
            self.lon_col = lon_col
            self.cluster_col = cluster_col
            self.random_state = random_state
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, tol=1e-6)

        def fit(self, X, y=None):
            self.kmeans.fit(X[[self.lat_col, self.lon_col]])
            self.feature_names_in_ = X.columns.tolist() 
            return self

        def transform(self, X, y=None):
            X = X.copy()
            X[self.cluster_col] = self.kmeans.predict(X[[self.lat_col, self.lon_col]])
            return X
class HousingPreprocessor:
    def __init__(self, pipeline_dir = PIPELINE_DIR,
                 data_dir = DATA_DIR):
        self.pipeline_path = os.path.join(pipeline_dir, 'housing_pipeline.joblib')
        self.x_train_path = os.path.join(data_dir, 'X_train.csv')
        self.x_test_path = os.path.join(data_dir, 'X_test.csv')
        self.y_train_path = os.path.join(data_dir, 'y_train.csv')
        self.y_test_path = os.path.join(data_dir, 'y_test.csv')
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_pipeline(self):
        try:
            self.pipeline = joblib.load(self.pipeline_path)
            return True
        except FileNotFoundError:
            return False

    def preprocess_array(self, array_data):
        if self.pipeline is None:
            if not self.load_pipeline():
                raise ValueError("Pipeline not loaded. Please train and save the pipeline first.")
        
        if not isinstance(array_data[0], list):
            array_data = [array_data]

        df = pd.DataFrame(array_data, columns=self.pipeline.feature_names_in_)
        preprocessed_data = self.pipeline.transform(df)
        return preprocessed_data

    def preprocess_api_json(self, api_json):
        if self.pipeline is None:
            if not self.load_pipeline():
                raise ValueError("Pipeline not loaded. Please train and save the pipeline first.")

        df = pd.DataFrame([api_json])
        preprocessed_data = self.pipeline.transform(df)
        return preprocessed_data

    def feature_engineering(self, X):
        X['rooms_per_bedroom'] = X['AveRooms'] / X['AveBedrms']
        X['rooms_per_population'] = X['AveRooms'] / X['Population']
        X['population_per_household'] = X['Population'] / X['AveOccup']
        return X
    
    

    def create_pipeline_and_process_data(self, X_train, X_test, y_train, y_test):
        # add cluster
        cluster_adder = ClusterAdder(n_clusters=5)

        # feature engineer
        feature_engineer = FunctionTransformer(self.feature_engineering)

        # numeric and categorical features
        numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
        categorical_features = ['location_cluster']

        #transformers
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('log_transform', QuantileTransformer(output_distribution='normal', n_quantiles=500)),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # column Transformer
        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        # pipeline
        self.pipeline = Pipeline(steps=[
            ('cluster_adder', cluster_adder),
            ('feature_engineer', feature_engineer),
            ('preprocessor', preprocessor)
        ])

        # fit and transform
        self.X_train, self.X_test, self.y_train, self.y_test = self.pipeline.fit_transform(X_train, y_train), self.pipeline.transform(X_test), y_train, y_test
        return self.X_train, self.X_test, self.y_train, self.y_test

    def save_dataset(self):
        pd.DataFrame(self.X_train).to_csv(self.x_train_path, index=False)
        pd.DataFrame(self.X_test).to_csv(self.x_test_path, index=False)
        pd.DataFrame(self.y_train).to_csv(self.y_train_path, index=False, header=['MedHouseVal'])
        pd.DataFrame(self.y_test).to_csv(self.y_test_path, index=False, header=['MedHouseVal'])

if __name__ == "__main__":
    housing = fetch_california_housing(as_frame=True)
    data = housing.frame

    X = data.drop('MedHouseVal', axis=1)
    y = data['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = HousingPreprocessor()

    if not preprocessor.load_pipeline():
        X_train, X_test, y_train, y_test = preprocessor.create_pipeline_and_process_data(X_train, X_test, y_train, y_test)
        joblib.dump(preprocessor.pipeline, preprocessor.pipeline_path)
        preprocessor.save_dataset()

        logger.info("Pipeline trained and saved.")
    else:
        logger.info("Loaded existing pipeline.")

    # Get original features as dictionary
    original_features = X.iloc[0].to_dict()  
    print(f"Original Features: {original_features}")

    # Convert the dict to a list
    original_features_list = list(original_features.values()) 

    # Use preprocess_array
    preprocessed_array = preprocessor.preprocess_array([original_features_list]) 
    print(f'Preprocessed array: {preprocessed_array}')

    api_json = {'AveBedrms': 1.0238095238095237, 'HouseAge': 41.0, 'MedInc': 8.3252,  'AveRooms': 6.984126984126984,  'Population': 322.0, 'AveOccup': 2.5555555555555554, 'Latitude': 37.88, 'Longitude': -122.23}
    preprocessed_data = preprocessor.preprocess_api_json(api_json)
    print(f"Preprocessed Data: {preprocessed_data}")