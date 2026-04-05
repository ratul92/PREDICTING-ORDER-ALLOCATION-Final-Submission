import joblib
import pandas as pd
import numpy as np
import os
import holidays
from textblob import TextBlob # Added for the new NLP sentiment feature

class SupplyChainInference:
    def __init__(self, models_path=r'A:\desktop\New folder\P\Ratul\Supply_Chain_Thesis\models'):
        """Load all exported artifacts from the models folder"""
        self.scaler = joblib.load(os.path.join(models_path, 'production_scaler.pkl'))
        self.encoder = joblib.load(os.path.join(models_path, 'production_encoder.pkl'))
        self.kmeans = joblib.load(os.path.join(models_path, 'production_kmeans.pkl'))
        self.tfidf = joblib.load(os.path.join(models_path, 'production_tfidf.pkl'))
        self.features_list = joblib.load(os.path.join(models_path, 'production_features_list.pkl'))
        self.model = joblib.load(os.path.join(models_path, 'production_stacked_model.pkl')) 
        self.us_holidays = holidays.US()
        
        # Static mapping for market frequencies based on DataCo dataset distributions
        self.market_freq_map = {
            'LATAM': 0.286, 'Europe': 0.278, 'Pacific Asia': 0.228, 'USCA': 0.142, 'Africa': 0.062
        }

    def preprocess(self, raw_data):
        """Transform raw input data into the exact format the final model expects"""
        df = raw_data.copy()
        
        # 1. Temporal & Holiday Features
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['order_month'] = df['order_date'].dt.month
        df['order_day_of_week'] = df['order_date'].dt.dayofweek
        df['is_weekend'] = df['order_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_holiday'] = df['order_date'].dt.date.apply(lambda x: 1 if x in self.us_holidays else 0)

        # 2. Geospatial Zones (50 Zones)
        df['geospatial_zone'] = self.kmeans.predict(df[['latitude', 'longitude']])

        # 3. Text Metrics: Sentiment & Readability (NEW)
        df['text_sentiment'] = df['category_name'].fillna('').apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['text_readability_length'] = df['category_name'].fillna('').apply(lambda x: len(str(x).split()))

        # 4. Target / Frequency Encoding (NEW)
        df['market_freq_encoded'] = df['market'].map(self.market_freq_map).fillna(0.2)

        # 5. 7-Day Rolling Average Proxy (NEW)
        # For a stateless, real-time API request, we impute with a baseline global average proxy
        df['zone_7d_rolling_avg'] = 3.5

        # 6. Capacity Proxies & Delivery Ratios (NEW)
        df['capacity_proxy_value_per_item'] = df['sales'] / df['order_item_quantity'].replace(0, 1)
        df['economic_delivery_ratio'] = df['order_item_discount'] / df['sales'].replace(0, 1)

        # 7. Scaling and Encoding
        num_cols = [
            'order_month', 'order_day_of_week', 'is_weekend', 'is_holiday', 
            'order_item_quantity', 'sales', 'order_item_discount',
            'text_sentiment', 'text_readability_length', 'market_freq_encoded', 'zone_7d_rolling_avg',
            'capacity_proxy_value_per_item', 'economic_delivery_ratio'
        ]
        cat_cols = ['shipping_mode', 'market', 'customer_segment', 'order_region', 'geospatial_zone']
        text_col = 'category_name'

        scaled_nums = pd.DataFrame(self.scaler.transform(df[num_cols]), columns=num_cols)
        encoded_cats = pd.DataFrame(self.encoder.transform(df[cat_cols]), columns=self.encoder.get_feature_names_out(cat_cols))
        
        # 8. TF-IDF Text Processing
        text_features = pd.DataFrame(self.tfidf.transform(df[text_col].fillna('')).toarray(), columns=[f"tfidf_{w}" for w in self.tfidf.get_feature_names_out()])
        
        # Combine and align with the training feature list
        final_df = pd.concat([scaled_nums, encoded_cats, text_features], axis=1)
        
        # Ensure columns are in the exact same order as training
        return final_df[self.features_list]

    def predict_lead_time(self, raw_data):
        """Generate final numerical prediction in days"""
        processed_data = self.preprocess(raw_data)
        prediction = self.model.predict(processed_data)
        return np.round(prediction, 2)