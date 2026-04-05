import requests
import time

# The address of your local API
url = "http://127.0.0.1:8000/predict"

# Sample data to send (Added category_name and updated coordinates)
payload = {
    "order_date": "2026-03-13",
    "latitude": 17.3850, 
    "longitude": 78.4867,
    "order_item_quantity": 1,
    "sales": 150.00,
    "order_item_discount": 15.0,
    "shipping_mode": "First Class",
    "market": "LATAM",
    "customer_segment": "Consumer",
    "order_region": "Central America",
    "category_name": "Sporting Goods" # New field required by the TF-IDF vectorizer
}

print("Testing Stacked Ensemble API Latency...")
try:
    start_time = time.perf_counter()
    response = requests.post(url, json=payload)
    end_time = time.perf_counter()
    
    if response.status_code == 200:
        result = response.json()
        latency = (end_time - start_time) * 1000 
        print(f"--- SUCCESS ---")
        print(f"Prediction: {result['predicted_lead_time_days']:.2f} days")
        print(f"Model Version: {result['model_version']}")
        print(f"API Response Time: {latency:.2f} ms")
    else:
        print(f"Server returned error {response.status_code}: {response.text}")
except Exception as e:
    print(f"Connection failed: {e}")