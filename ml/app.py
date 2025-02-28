from flask import Flask, request, jsonify
import joblib
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
linearmodel = joblib.load("linearreg_model.pkl")
mulmodel = joblib.load("multilinearreg_model.pkl")
#scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Request received")
    data = request.get_json()
    features = np.array([data['Square_Feet'], data['Bedrooms']]).reshape(1, -1)

    # Scale the input
    #features_scaled = scaler.transform(features)
    logging.info(features)
    # Predict
    prediction = linearmodel.predict(features)[0]
    logging.info(prediction)
    return jsonify({'Predicted Price': prediction})

@app.route('/predictmulti', methods=['POST'])
def predictmulti():
    data = request.get_json()
    features = np.array([data['Square_Feet'], data['Bedrooms'],
    data['Bathrooms'], data['Proximity_to_City_Center'],
    data['Age_of_House']]).reshape(1, -1)
    logging.info(features)
    # Scale the input
    #features_scaled = scaler.transform(features)

    # Predict
    prediction = mulmodel.predict(features)[0]
    logging.info(prediction)
    return jsonify({'Predicted Price': prediction})

if __name__ == '__main__':
    app.run(debug=True)