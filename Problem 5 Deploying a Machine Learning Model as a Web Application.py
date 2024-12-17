# Module 5: Machine Learning Model Deployment
# Problem 5: Deploying a Machine Learning Model as a Web Application
# Objective: Deploy a trained machine learning model (from any of the previous problems) to a web application using Flask.
"""
Tasks:
•	Train a machine learning model on a chosen dataset.
•	Create a Flask API to serve the model for predictions.
•	Host the application on platforms like Heroku or AWS.
"""

# Source Code (Flask API):
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model (assume model is saved as 'model.pkl')
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify(prediction=prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
"""
Outcome: Students will learn how to deploy a machine learning model as a 
web service and integrate it with real-world applications.
"""