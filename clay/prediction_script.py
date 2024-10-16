# prediction_script.py

import sys
import os
import math
import geopandas as gpd
import numpy as np
import pandas as pd
import pystac_client
import stackstac
import torch
import yaml
from box import Box
from shapely.geometry import Point
from sklearn import svm
from torchvision.transforms import v2
import joblib
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
import json
from datetime import datetime
from geopy.geocoders import Nominatim
from dateutil.parser import parse as date_parse

# Import the Clay model
sys.path.append("../..")  # Adjust the path to your project's root if necessary
from src.model import ClayMAEModule

def main():
    # Load the trained classifier from disk
    clf = joblib.load('svm_classifier.joblib')
    print("Classifier loaded from 'svm_classifier.joblib'.")

    # Load the Clay model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt = "https://clay-model-ckpt.s3.amazonaws.com/v0.5.7/mae_v0.5.7_epoch-13_val-loss-0.3098.ckpt"

    model = ClayMAEModule.load_from_checkpoint(
        ckpt, metadata_path="../../configs/metadata.yaml", shuffle=False, mask_ratio=0
    )
    model.eval()
    model = model.to(device)

    print("Loaded Model")

    # Define constants
    size = 256
    gsd = 10  # Ground Sample Distance in meters
    platform = "sentinel-2-l2a"

    # Prepare band metadata
    metadata = Box(yaml.safe_load(open("../../configs/metadata.yaml")))
    mean = []
    std = []
    waves = []
    band_list = ["blue", "green", "red", "nir"]
    for band_name in band_list:
        mean.append(metadata[platform].bands.mean[band_name])
        std.append(metadata[platform].bands.std[band_name])
        waves.append(metadata[platform].bands.wavelength[band_name])

    transform = v2.Compose([v2.Normalize(mean=mean, std=std)])

    # Read extracted parameters from JSON file
    extracted_params_file = 'extracted_params.json'
    try:
        with open(extracted_params_file, 'r') as f:
            extracted_params = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{extracted_params_file}' not found. Please run the backend code first.")
        return

    # Extract location name and date range
    location_name = extracted_params.get('location_name')
    start_date_str = extracted_params.get('start_date')
    end_date_str = extracted_params.get('end_date')

    if not all([location_name, start_date_str, end_date_str]):
        print("Error: Missing required parameters in extracted_params.json.")
        return

    # Geocode the location name to get coordinates
    geolocator = Nominatim(user_agent="clay_model_demo (your_email@example.com)")
    location = geolocator.geocode(location_name)
    if location is None:
        print(f"Error: Could not geocode location: {location_name}")
        return

    latitude = location.latitude
    longitude = location.longitude
    location_coords = (latitude, longitude)

    # Parse dates
    start_date = date_parse(start_date_str).date()
    end_date = date_parse(end_date_str).date()

    print(f"Using location: {location_name} ({latitude}, {longitude})")
    print(f"Using date range: {start_date} to {end_date}")

    # Output directories
    output_image_dir = "output_images"
    output_metadata_file = "output_metadata.json"

    # Create the output directory if it doesn't exist
    os.makedirs(output_image_dir, exist_ok=True)

    # Fetch new data
    new_items = fetch_new_data(location_coords, str(start_date), str(end_date))
    print(f"Found {len(new_items)} new items for prediction.")

    if len(new_items) == 0:
        print("No new data available for the specified location and date range.")
    else:
        # Generate embeddings and collect images
        new_embeddings, images = generate_new_embeddings(
            new_items, location_coords, model, transform, platform, device, waves, size, gsd
        )

        # Make predictions
        new_predictions = predict_new_data(clf, new_embeddings)

        # Prepare metadata list
        metadata_list = []

        # Output results and save images
        for i, prediction in enumerate(new_predictions):
            item_date = new_items[i].datetime.date()
            stack = images[i]

            # Extract RGB image and remove singleton dimensions
            rgb_image = stack.sel(band=["red", "green", "blue"]).isel(time=0)

            # Ensure dimensions are (y, x, band)
            rgb_image = rgb_image.transpose("y", "x", "band")

            # Convert to numpy array
            rgb_array = rgb_image.values

            # Handle missing or invalid data
            if np.all(np.isnan(rgb_array)):
                print(f"No valid data available for {item_date}. Skipping image.")
                continue

            # Clip values and normalize
            rgb_array = np.clip(rgb_array, 0, 2000) / 2000  # Normalize to [0, 1]

            # Save the image to disk
            image_filename = f"image_{i}_{item_date}.png"
            image_path = os.path.join(output_image_dir, image_filename)
            plt.imsave(image_path, rgb_array)
            plt.close()

            # Collect metadata
            metadata = {
                "image_filename": image_filename,
                "date": str(item_date),
                "location": {
                    "name": location_name,
                    "latitude": latitude,
                    "longitude": longitude
                },
                "prediction": int(prediction),  # Convert to int for JSON serialization
                "prediction_label": "Forest fire detected" if prediction == 2 else "No forest fire detected"
            }
            metadata_list.append(metadata)

        # Save metadata to JSON file
        with open(output_metadata_file, 'w') as f:
            json.dump(metadata_list, f, indent=4)
        print(f"Metadata saved to '{output_metadata_file}'.")

# Define normalization functions
def normalize_timestamp(date):
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24
    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

def normalize_latlon(lat, lon):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))

# Function to fetch new data
def fetch_new_data(location, start_date, end_date):
    lat, lon = location

    # Search STAC catalog for new data
    STAC_API = "https://earth-search.aws.element84.com/v1"
    COLLECTION = "sentinel-2-l2a"

    catalog = pystac_client.Client.open(STAC_API)
    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{start_date}/{end_date}",
        bbox=(lon - 0.0001, lat - 0.0001, lon + 0.0001, lat + 0.0001),
        max_items=100,
        query={"eo:cloud_cover": {"lt": 80}},
    )

    return search.get_all_items()

# Function to generate embeddings and collect images
def generate_new_embeddings(new_items, location, model, transform, platform, device, waves, size, gsd):
    new_embeddings = []
    images = []

    lat, lon = location

    for item in new_items:
        # Extract EPSG and bounds from the current item
        epsg = item.properties["proj:epsg"]

        # Convert point of interest into the image projection
        poidf = gpd.GeoDataFrame(
            pd.DataFrame(),
            crs="EPSG:4326",
            geometry=[Point(lon, lat)],
        ).to_crs(epsg)
        coords = poidf.iloc[0].geometry.coords[0]

        # Create bounds
        bounds = (
            coords[0] - (size * gsd) // 2,
            coords[1] - (size * gsd) // 2,
            coords[0] + (size * gsd) // 2,
            coords[1] + (size * gsd) // 2,
        )

        # Process new imagery
        stack = stackstac.stack(
            [item],
            bounds=bounds,
            snap_bounds=False,
            epsg=epsg,
            resolution=gsd,
            assets=["blue", "green", "red", "nir"],
            resampling=Resampling.nearest,
            dtype="float32",
            fill_value=np.float32(0),
            rescale=False,
        )
        stack = stack.compute()

        # Save the image data
        images.append(stack)

        # Prepare time embedding
        datetimes = stack.time.values.astype("datetime64[s]").tolist()
        times = [normalize_timestamp(dat) for dat in datetimes]
        week_norm = [t[0] for t in times]
        hour_norm = [t[1] for t in times]

        # Prepare lat/lon embedding
        latlons = [normalize_latlon(lat, lon)] * len(times)
        lat_norm = [ll[0] for ll in latlons]
        lon_norm = [ll[1] for ll in latlons]

        # Normalize pixels
        pixels = torch.from_numpy(stack.data.astype(np.float32))
        pixels = transform(pixels)

        # Prepare the data cube
        datacube = {
            "platform": platform,
            "time": torch.tensor(np.hstack((week_norm, hour_norm)), dtype=torch.float32, device=device),
            "latlon": torch.tensor(np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=device),
            "pixels": pixels.to(device),
            "gsd": torch.tensor(stack.gsd.values, device=device),
            "waves": torch.tensor(waves, device=device),
        }

        with torch.no_grad():
            # Generate embeddings
            unmsk_patch, *_ = model.model.encoder(datacube)
            embedding = unmsk_patch[:, 0, :].cpu().numpy()
            new_embeddings.append(embedding)

    # Concatenate embeddings
    new_embeddings = np.concatenate(new_embeddings, axis=0)
    return new_embeddings, images

# Function to make predictions
def predict_new_data(clf, new_embeddings):
    predictions = clf.predict(new_embeddings + 100)
    return predictions

if __name__ == "__main__":
    main()
