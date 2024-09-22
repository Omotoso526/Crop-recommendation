import joblib
import numpy as np

# Load the saved model
model = joblib.load('crop_recommendation_model.joblib')

def predict_crop(features: list):
    """
    Takes a list of features and returns the predicted crop recommendation.
    """
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return prediction[0]
