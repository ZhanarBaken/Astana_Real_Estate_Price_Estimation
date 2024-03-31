# External libraries
import os
import datetime
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory
from geopy.distance import geodesic
import nbimporter
import pickle
import re
import ast


# Web scraping and API libraries
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Statistical analysis and modeling
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Importing necessary functions from data_mining module
from data_mining import geocode_2gis, count_places_within_radius, checking_park

# Importing the data_processing_pipeline function from the main module
from main import data_processing_pipeline

# Committing RANDOM_SEED to make experiments repeatable
SEED = 42


# Create flask app
flask_app = Flask(__name__)

# Function to check if model update is needed
def check_model_update():
    # Path to the file storing the information about the last model update time
    model_update_file = "model_last_update.txt"
    
    # Check if the file exists
    if not os.path.exists(model_update_file):
        # If the file doesn't exist, create it and write the current date
        with open(model_update_file, "w") as f:
            f.write(str(datetime.datetime.now()))
        return True  # Return True to load the model on the first run
    
    # Read the last update date from the file
    with open(model_update_file, "r") as f:
        last_update_str = f.read().strip()
    last_update = datetime.datetime.strptime(last_update_str, "%Y-%m-%d %H:%M:%S.%f")
    
    # Check if a year has passed since the last model update
    current_time = datetime.datetime.now()
    if (current_time - last_update).days >= 365:
        # If a year has passed, return True to update the model
        return True
    else:
        # Otherwise, return False to not update the model
        return False

# Function to update the model
def update_model():
    # Load the existing model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
        
    # Call the function to prepare data
    X_train_new, y_train_new, = data_processing_pipeline()
    
    # Retrain the model
    model.fit(X_train_new, y_train_new)
    
    # Save the updated model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Save the model update date
    model_update_file = "model_last_update.txt"
    with open(model_update_file, "w") as f:
        f.write(str(datetime.datetime.now()))
    
    return model



# Function to process data and to create new features 
def process_data(data):
    columns = [
    'owner', 'complex_name', 'house_type', 'in_pledge', 'construction_year',
    'ceiling_height', 'bathroom_info', 'condition', 'area', 'room_count',
    'floor', 'floor_count', 'district', 'complex_class', 'parking',
    'elevator', 'schools_within_500m', 'kindergartens_within_500m',
    'park_within_1km', 'distance_to_center', 'distance_to_botanical_garden',
    'distance_to_triathlon_park', 'distance_to_astana_park',
    'distance_to_treatment_facility', 'distance_to_railway_station_1',
    'distance_to_railway_station_2', 'distance_to_industrial_zone',
    'last_floor', 'first_floor'
    ]
    
    owner = data["owner"]        
    complex_name = data["complex_name"]
    house_type = data["house_type"]
    in_pledge = data["in_pledge"] == 'yes'
    construction_year = int(data["construction_year"])
    ceiling_height = float(data["ceiling_height"])
    ceiling_height = min([2.5, 2.7, 2.8, 2.9, 3, 3.3, 3.5, 4], key=lambda x: abs(x - ceiling_height))
    bathroom_info = data["bathroom_info"]
    condition = data["condition"]
    area = float(data["area"])
    room_count = int(data["room_count"])
    floor = int(data["floor"])
    floor_count = int(data["floor_count"])  
    district = data["district"]
    complex_class = data["complex_class"]
    parking = data["parking"]
    elevator = data["elevator"]
    
    address = data["address"]
    coordinates_str = geocode_2gis(address)
    coordinates_list = coordinates_str.replace('(', '').replace(')', '') 
    latitude = float(coordinates_list[0])
    longitude = float(coordinates_list[1])
    coordinates = (latitude, longitude)
    
    schools_within_500m = float(count_places_within_radius("school", coordinates))
    schools_within_500m = min(schools_within_500m, 4)
    
    kindergartens_within_500m = float(count_places_within_radius("kindergarten", coordinates))
    kindergartens_within_500m = min(kindergartens_within_500m, 3)
    
    park_within_1km = checking_park(coordinates)
    geo_center_of_astana = (51.128318, 71.430381)
    distance_to_center = geodesic(geo_center_of_astana, coordinates).kilometers
    botanical_garden = (51.106433, 71.416329)
    distance_to_botanical_garden = geodesic(botanical_garden, coordinates).kilometers
    triathlon_park = (51.13593, 71.449809)
    distance_to_triathlon_park =  geodesic(triathlon_park, coordinates).kilometers
    astana_park = (51.156264, 71.419961)
    distance_to_astana_park = geodesic(astana_park, coordinates).kilometers
    treatment_facility = (51.144302, 71.337247)
    distance_to_treatment_facility = geodesic(treatment_facility, coordinates).kilometers
    railway_station_1 = (51.195572, 71.409173)
    distance_to_railway_station_1 = geodesic(railway_station_1, coordinates).kilometers
    railway_station_2 = (51.112488, 71.531596)
    distance_to_railway_station_2 = geodesic(railway_station_2, coordinates).kilometers
    industrial_zone = (51.140231, 71.551219)
    distance_to_industrial_zone = geodesic(industrial_zone, coordinates).kilometers
    
    last_floor = floor == floor_count
    first_floor = floor == 1 or (floor == 2 and parking == 'underground')

    data_for_df = [
        owner, complex_name, house_type, in_pledge, construction_year, ceiling_height, bathroom_info, 
        condition, area, room_count, floor, floor_count, district, complex_class, parking, elevator, 
        schools_within_500m, kindergartens_within_500m, park_within_1km, distance_to_center, 
        distance_to_botanical_garden, distance_to_triathlon_park, distance_to_astana_park, 
        distance_to_treatment_facility, distance_to_railway_station_1, distance_to_railway_station_2, 
        distance_to_industrial_zone, last_floor, first_floor
    ]
    
    processed_df = pd.DataFrame([data_for_df], columns=columns)
    return processed_df


# Check if model update is needed
if check_model_update():
    # If update is needed, update the model
    model = update_model()
else:
    # If no update is needed, load the existing model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

@flask_app.route('/data/<path:path>')
def send_static(path):
    return send_from_directory('data', path)
        
@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route('/favicon.ico')
def favicon():
    return flask_app.send_static_file('favicon.ico')

@flask_app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()
    processed_data = process_data(form_data)
    prediction = model.predict(processed_data)
    prediction = np.exp(prediction)
    prediction_text = "Примерная стоимость: {:,.0f} тенге".format(prediction[0])
    return redirect(url_for('prediction_result', prediction_text=prediction_text))

@flask_app.route("/prediction_result")
def prediction_result():
    prediction_text = request.args.get('prediction_text', '')
    return render_template("prediction_result.html", prediction_text=prediction_text)

if __name__ == "__main__":
    flask_app.run(debug=True, port=8080)