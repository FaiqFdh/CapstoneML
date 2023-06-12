import os
import uvicorn
import traceback
import tensorflow as tf
import pandas as pd
import random
import numpy as np

from pydantic import BaseModel
from urllib.request import Request
from fastapi import FastAPI, Response, UploadFile,Query
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

#Load Model
model = tf.keras.models.load_model('./recommendation_rating_model.h5')

#Initialize fastAPI
app = FastAPI()

# This endpoint is for a test to this server
@app.get("/")
def index():
    return "Hello world from ML endpoint!"

class RequestPredict(BaseModel):
    user_id: str

tourism = pd.read_csv('./tourism_data.csv')
rating = pd.read_csv('./user_rating.csv')

#TIDAK BOLEH ADA VALUE KOSONG , JIKA ADA ERROR RETURN JSON

@app.post("/predict")
def predict(req : RequestPredict, response: Response):
#def predict(response: Response, user_id : int = Query(...)):
    try:
        user_id = req.user_id
        
        if user_id in rating['user_id'].values:
            
            # Convert user ID to integer
            user_id = pd.Series([user_id]).astype('category').cat.codes.values[0]

            # Create input data for recommendations
            user_data = np.array([user_id] * len(tourism['id'].unique()))
            tourism_data = np.array(list(tourism['id'].unique()))

            # Make predictions
            predictions = model.predict([user_data, tourism_data]).flatten()

            top_k=10
            # Get top-k recommendations
            top_indices = predictions.argsort()[-top_k:][::-1]
            top_recommendations = tourism.iloc[top_indices]['id']
            
            # Convert recommended_tourism_ids to a pandas Series
            top_recommendations_series = pd.Series(top_recommendations)

            # Filter the rows in tempat that have the same place IDs as recommended_tourism_ids
            recommend = tourism[tourism['id'].isin(top_recommendations_series)]
                        
            return {"recommendations": recommend.to_dict(orient='records')}

        else:
            # User ID doesn't exist, make random recommendations
             # Get the total number of tourism data
            print("Random Rekomendasi")
            total_tourism = len(tourism)

            # Generate random indices to select random recommendations
            num_recommendations = 5
            random_indices = random.sample(range(len(tourism)), num_recommendations)

            # Get the random recommendations based on the selected indices
            recommend = tourism.iloc[random_indices]

            return {"recommendations": recommend.to_dict(orient='records')}
                    
     
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"

#Nearest Recommendations Using K-Means
#step 1 read the file
#tourism = pd.read_csv('./tourism_data.csv')

# Step 2: Preprocess the data (if needed)

# Step 3: Feature engineering
features = tourism[['lat', 'lon']]

# Step 4: Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 5: Apply k-means clustering
k = 4  # Set the number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_features)

# Step 6: Assign cluster labels
cluster_labels = kmeans.labels_

class RequestLoc(BaseModel):
    latitude: float
    longitude: float
       
# latitude = -6.175392	
# longitude = 106.827153

@app.post("/predict_loc")
def recommend_locations(req:RequestLoc, response: Response):

    # Function implementation
    try:
        latitude = req.latitude
        longitude = req.longitude
        target_location = [latitude, longitude]
        target_scaled = scaler.transform([target_location])

        # Find the nearest neighbors
        n_neighbors = 10  # Set the number of nearest neighbors to recommend
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(scaled_features)
        distances, indices = nbrs.kneighbors(target_scaled)

        # Get the recommended locations based on cluster labels
        recommended_locations = tourism.iloc[indices[0]]
        
        return recommended_locations.to_dict(orient='records')

    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"


# Starting the server
# Your can check the API documentation easily using /docs after the server is running
port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0',port=port)
