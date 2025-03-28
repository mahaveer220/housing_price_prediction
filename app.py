import numpy as np
from flask import Flask, request, render_template
import joblib
from logger import Logger


import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"src")))
print(sys.path)
from preprocessing import HousingPreprocessor,ClusterAdder




app = Flask(__name__)
model_path = "/Users/mahaveer/Desktop/soulai/housing_price_prediction/src/best_model.pkl"
model = joblib.load(model_path)
logger = Logger()

preprocessor= HousingPreprocessor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form values and convert them to float
        input_features = [float(request.form[field]) for field in [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"
        ]]
        
        # Convert input into model-friendly format
        final_features = np.array(input_features)
        preprocessed_array = preprocessor.preprocess_array(final_features) 
        
        # Get prediction
        prediction = model.predict(preprocessed_array)
        output = round(prediction[0], 2)

        logger.info(f'Predictions run on {input_features}, result: {output}')
        return render_template('index.html', prediction_text=f'Predicted House Price: ${output}')
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return render_template('index.html', prediction_text="Error: Invalid Input")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
