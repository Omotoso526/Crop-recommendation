from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib

# Create the FastAPI instance
app = FastAPI()

@app.get('/')
def home():
    return{'crop': 'Croptype prediction'}

# Define the data structure for input
class CropFeatures(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Define the prediction endpoint
@app.post("/predict")
def get_prediction(nitrogen: float,
    phosphorus: float,
    potassium: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float):
    

    # Get prediction from model
    
    model = joblib.load('crop_recommendation_model.joblib1')
    prediction = model.predict([[nitrogen,phosphorus,potassium,temperature,humidity,ph,rainfall]])
    crop_map = {0: 'rice', 1: 'maize', 2: 'chickpea', 3: 'kidneybeans', 4: 'pigeonpeas', 
            5: 'mothbeans', 6: 'mungbean', 7: 'blackgram', 8: 'lentil', 9: 'pomegranate', 
            10: 'banana', 11: 'mango', 12: 'grapes', 13: 'watermelon', 14: 'muskmelon', 
            15: 'apple', 16: 'orange', 17: 'papaya', 18: 'coconut', 19: 'cotton', 
            20: 'jute', 21: 'coffee'}

    return {"prediction": crop_map[int(prediction[0])]}

if __name__ == '__main__':
    uvicorn.run(app)
