from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from inference import SupplyChainInference
import uvicorn

# Initialize the FastAPI app
app = FastAPI(title="Supply Chain Lead Time Predictor (Stacked Ensemble)")

# Load the updated inference engine
engine = SupplyChainInference()

# Define the updated input data structure (Added category_name)
class ShipmentData(BaseModel):
    order_date: str
    latitude: float
    longitude: float
    order_item_quantity: int
    sales: float
    order_item_discount: float
    shipping_mode: str
    market: str
    customer_segment: str
    order_region: str
    category_name: str  # New field required by the TF-IDF vectorizer

@app.get("/")
def home():
    return {"message": "Supply Chain Prediction API (Stacked Ensemble) is Running Locally"}

@app.post("/predict")
def predict(data: ShipmentData):
    # Convert input data to DataFrame using model_dump() or dict()
    input_df = pd.DataFrame([data.dict()])
    
    # Generate prediction using the inference engine
    prediction = engine.predict_lead_time(input_df)
    
    return {
        "predicted_lead_time_days": float(prediction[0]),
        "model_version": "Stacked Ensemble (XGBoost + LightGBM + Ridge)",
        "status": "success"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)