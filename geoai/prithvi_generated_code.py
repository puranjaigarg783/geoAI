# Import necessary libraries
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from datasets import load_dataset
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import os

# Load the fine-tuned Prithvi model
model = AutoModelForImageClassification.from_pretrained('models/pool_detector_model')
feature_extractor = AutoFeatureExtractor.from_pretrained('models/pool_detector_model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load satellite imagery data for Los Angeles
dataset = load_dataset('your_dataset_name', split='test')  # Replace with actual dataset

# Process images and perform pool detection
pool_locations = []
for example in dataset:
    image = example['image']
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    if prediction == 1:  # Assuming label 1 indicates a pool
        # Extract geolocation metadata
        lat, lon = example['latitude'], example['longitude']
        pool_locations.append({'geometry': Point(lon, lat)})

# Create GeoDataFrame
pool_locations_gdf = gpd.GeoDataFrame(pool_locations, crs="EPSG:4326")

# Save pool locations to a file
pool_locations_gdf.to_file('output/pool_locations.geojson', driver='GeoJSON')

# Create a map
m = folium.Map(location=[34.05, -118.25], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)
for idx, row in pool_locations_gdf.iterrows():
    folium.Marker(location=[row.geometry.y, row.geometry.x]).add_to(marker_cluster)

# Save the map
map_filepath = 'output/pools_map.html'
m.save(map_filepath)
print(f"Map saved to: {map_filepath}")

