# README
# Hello everyone, in here I (Kaenova | Bangkit Mentor ML-20) 
# will give you some headstart on createing ML API. 
# Please read every lines and comments carefully. 
# 
# I give you a headstart on text based input and image based input API. 
# To run this server, don't forget to install all the libraries in the
# requirements.txt simply by "pip install -r requirements.txt" 
# and then use "python main.py" to run it
# 
# For ML:
# Please prepare your model either in .h5 or saved model format.
# Put your model in the same folder as this main.py file.
# You will load your model down the line into this code. 
# There are 2 option I give you, either your model image based input 
# or text based input. You need to finish functions "def predict_text" or "def predict_image"
# 
# For CC:
# You can check the endpoint that ML being used, eiter it's /predict_text or 
# /predict_image. For /predict_text you need a JSON {"text": "your text"},
# and for /predict_image you need to send an multipart-form with a "uploaded_file" 
# field. you can see this api documentation when running this server and go into /docs
# I also prepared the Dockerfile so you can easily modify and create a container iamge
# The default port is 8080, but you can inject PORT environement variable.
# 
# If you want to have consultation with me
# just chat me through Discord (kaenova#2859) and arrange the consultation time
#
# Share your capstone application with me! 🥳
# Instagram @kaenovama
# Twitter @kaenovama
# LinkedIn /in/kaenova

## Start your code here! ##

import os
import uvicorn
import traceback
import tensorflow as tf
import pandas as pd

from pydantic import BaseModel
from urllib.request import Request
from fastapi import FastAPI, Response, UploadFile,Query

import numpy as np
#from utils import load_image_into_numpy_array

# Initialize Model
# If you already put yout model in the same folder as this main.py
# You can load .h5 model or any model below this line

# If you use h5 type uncomment line below
model = tf.keras.models.load_model('./recommendation_rating_model.h5')
# If you use saved model type uncomment line below
# model = tf.saved_model.load("./my_model_folder")

app = FastAPI()

# This endpoint is for a test (or health check) to this server
@app.get("/")
def index():
    return "Hello world from ML endpoint!"

# If your model need text input use this endpoint!
# class RequestText(BaseModel):
#     text:str

class RequestPredict(BaseModel):
    #user_id: int
    user_id: str

# @app.post("/predict_text")
# def predict_text(req: RequestText, response: Response):
#     try:
#         # In here you will get text sent by the user
#         text = req.text
#         print("Uploaded text:", text)
        
#         # Step 1: (Optional) Do your text preprocessing
        
#         # Step 2: Prepare your data to your model
        
#         # Step 3: Predict the data
#         # result = model.predict(...)
        
#         # Step 4: Change the result your determined API output
        
#         return "Endpoint not implemented"
#     except Exception as e:
#         traceback.print_exc()
#         response.status_code = 500
#         return "Internal Server Error"
tourism = pd.read_csv('./tourism_data.csv')
#rating = pd.read_csv('./tourism_rating.csv')
rating = pd.read_csv('./user_rating.csv')

tourism2 = pd.read_csv('./tourism_with_id.csv')
tourism2 = tourism2.drop(['Time_Minutes','Coordinate', 'Price' ,'Unnamed: 11', 'Unnamed: 12'], axis=1)
#TIDAK BOLEH ADA VALUE KOSONG , JIKA ADA ERROR RETURN JSON

import random

@app.post("/predict")
def predict(req : RequestPredict, response: Response):
#def predict(response: Response, user_id : int = Query(...)):
    try:
        user_id = req.user_id
        
        #jumlah id tempat
        #n_tourisms = len(tourism.Place_Id.unique())
        if user_id in rating['user_id'].values:
#             id_place = range(1, 436)
            
#             tourism_data = np.array(list(set(rating.Place_Id)))
#             user = np.array([user_id for i in range(len(tourism_data))])
  
#             predictions = model.predict([user, tourism_data])
#             predictions = np.array([a[0] for a in predictions])

#             recommended_tourism_ids = (-predictions).argsort()[:10]
            
#             # Convert recommended_tourism_ids to a pandas Series
#             recommended_tourism_ids_series = pd.Series(recommended_tourism_ids)

#             # Filter the rows in tempat that have the same place IDs as recommended_tourism_ids
#             filtered_tempat = tourism2[tourism2['Place_Id'].isin(recommended_tourism_ids_series)]
            
#             # Convert float values to strings
#             #filtered_tempat = filtered_tempat.astype(str)
            
#             #return {"recommended_tourism_ids": filtered_tempat}
#             return {"recommended_tourism_ids": filtered_tempat.to_dict(orient='records')}
        
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

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

#step 1 read the file
#tourism = pd.read_csv('./tourism_data.csv')
# Step 2: Preprocess the data (if needed)

# Step 3: Feature engineering
#features = tourism[['Lat', 'Long']]
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
#def recommend_locations(response: Response, latitude: float = Query(...), longitude: float = Query(...)):

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
        #recommended_locations = tourism.iloc[indices[0]][['Place_Id', 'Place_Name', 'Lat', 'Long']]
        recommended_locations = tourism.iloc[indices[0]]
        
        return recommended_locations.to_dict(orient='records')

    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"

    
# If your model need image input use this endpoint!
# @app.post("/predict_image")
# def predict_image(uploaded_file: UploadFile, response: Response):
#     try:
#         # Checking if it's an image
#         if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
#             response.status_code = 400
#             return "File is Not an Image"
        
#         # In here you will get a numpy array in "image" variable.
#         # You can use this file, to load and do processing
#         # later down the line
#         image = load_image_into_numpy_array(uploaded_file.file.read())
#         print("Image shape:", image.shape)
        
#         # Step 1: (Optional, but you should have one) Do your image preprocessing
        
#         # Step 2: Prepare your data to your model
        
#         # Step 3: Predict the data
#         # result = model.predict(...)
        
#         # Step 4: Change the result your determined API output
        
#         return "Endpoint not implemented"
#     except Exception as e:
#         traceback.print_exc()
#         response.status_code = 500
#         return "Internal Server Error"


# Starting the server
# Your can check the API documentation easily using /docs after the server is running
port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0',port=port)
