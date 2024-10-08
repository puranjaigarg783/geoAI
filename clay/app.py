import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import xarray as xr
import dask.array as da
import matplotlib.pyplot as plt
import torch
import sys
import torch.nn as nn
from torchvision import transforms
from shapely.geometry import box
from pystac_client import Client
import stackstac
from pyproj import Transformer
from planetary_computer import sign
from model.src.model import ClayMAEModule


# California bounding box: [west, south, east, north]
bbox = [-122.5, 37.7, -122.3, 37.8]  # California boundaries

# Define dates
today = datetime.datetime.utcnow().date()
two_weeks_ago = today - datetime.timedelta(days=14)

# Format dates
start_date = two_weeks_ago.strftime('%Y-%m-%d')
end_date = today.strftime('%Y-%m-%d')


print(f"Start Date: {start_date}")
print(f"End Date: {end_date}")

# Connect to the MPC STAC API
catalog = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')

# Search for Sentinel-2 L2A data
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=f"{start_date}/{end_date}",
    query={"eo:cloud_cover": {"lt": 20}},  # Less than 20% cloud cover
)

items = list(search.get_items())
print(f"Found {len(items)} Sentinel-2 items.")


# Function to filter items by date
def filter_items_by_date(items, target_date):
    target_date = pd.to_datetime(target_date).date()  # Convert to date object
    items_by_date = {}
    for item in items:
        item_date = pd.to_datetime(item.properties['datetime']).date()
        if item_date == target_date:
            items_by_date[item_date] = item
    if not items_by_date:
        # Find the item with the closest date
        closest_item = min(items, key=lambda item: abs(pd.to_datetime(item.properties['datetime']).date() - target_date))
        closest_date = pd.to_datetime(closest_item.properties['datetime']).date()
        print(f"No items on {target_date}, using closest date {closest_date}")
        return [closest_item]
    else:
        return list(items_by_date.values())



# Get items for current date
current_items = filter_items_by_date(items, today)

# Get items for two weeks ago
two_weeks_items = filter_items_by_date(items, two_weeks_ago)


# Define bands to use
bands = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR

# Sign the items (required for MPC data)
current_items = [sign(item) for item in current_items]
two_weeks_items = [sign(item) for item in two_weeks_items]


# Create image stack for current date
current_stack = stackstac.stack(
    current_items,
    assets=bands,
    bounds_latlon=bbox,
    resolution=10,
    chunksize=2048,
    dtype='float',
)

# Create image stack for two weeks ago
two_weeks_stack = stackstac.stack(
    two_weeks_items,
    assets=bands,
    bounds_latlon=bbox,
    resolution=10,
    chunksize=2048,
    dtype='float',
)


print(current_stack)
print(two_weeks_stack)

# Placeholder mean and std values for bands B02, B03, B04, B08
mean = np.array([1369.03, 1597.68, 1741.10, 2858.43], dtype='float32')
std = np.array([2026.96, 2011.88, 2146.35, 2016.38], dtype='float32')

# Function to normalize the data
def normalize(stack, mean, std):
    # stack: xarray DataArray with dimensions (time, band, y, x)
    # Convert mean and std to xarray DataArray for broadcasting
    mean_da = xr.DataArray(mean, dims=["band"], coords={"band": stack.band})
    std_da = xr.DataArray(std, dims=["band"], coords={"band": stack.band})
    normalized = (stack - mean_da) / std_da
    return normalized


current_stack_normalized = normalize(current_stack, mean, std)

two_weeks_stack_normalized = normalize(two_weeks_stack, mean, std)


def stack_to_tensor(stack):
    # Convert xarray DataArray to numpy array and then to PyTorch tensor
    np_stack = stack.data.compute().astype('float32')  # Ensure data is loaded into memory
    tensor = torch.from_numpy(np_stack)
    return tensor

# Convert and normalize current data
current_tensor = stack_to_tensor(current_stack_normalized)

# Convert and normalize two weeks ago data
two_weeks_tensor = stack_to_tensor(two_weeks_stack_normalized)

print("Current tensor shape:", current_tensor.shape)
print("Two weeks ago tensor shape:", two_weeks_tensor.shape)

def split_into_chips(tensor, chip_size=256):
    # tensor shape: (time, bands, height, width)
    _, _, height, width = tensor.shape
    # Calculate number of chips in each dimension
    num_chips_y = height // chip_size
    num_chips_x = width // chip_size
    # Split into chips
    chips = tensor.unfold(2, chip_size, chip_size).unfold(3, chip_size, chip_size)
    # Reshape to (num_chips_total, bands, chip_size, chip_size)
    chips = chips.contiguous().view(-1, tensor.size(1), chip_size, chip_size)
    return chips

# Split current tensor into chips
current_chips = split_into_chips(current_tensor)

# Split two weeks ago tensor into chips
two_weeks_chips = split_into_chips(two_weeks_tensor)

print("Number of current chips:", current_chips.shape[0])
print("Number of two weeks ago chips:", two_weeks_chips.shape[0])

# Path to the pre-trained model checkpoint
ckpt_path = 'clay_models/Clay_v0.1_epoch-24_val-loss-0.46.ckpt'  # Replace with actual path

ckpt = "https://clay-model-ckpt.s3.amazonaws.com/v0.5.7/mae_v0.5.7_epoch-13_val-loss-0.3098.ckpt"

# Load the model
model = ClayMAEModule.load_from_checkpoint(
    ckpt,
    metadata_path='model/configs/metadata.yaml',  # Adjust path as needed
    shuffle=False,
    mask_ratio=0  # No masking during inference
)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def normalize_timestamp(date):
    # Normalize week of the year
    week_of_year = date.isocalendar()[1]
    week_angle = week_of_year / 52 * 2 * np.pi
    week_norm = [np.sin(week_angle), np.cos(week_angle)]

    # Normalize hour of the day (assuming 0 hour for satellite images)
    hour_of_day = 0
    hour_angle = hour_of_day / 24 * 2 * np.pi
    hour_norm = [np.sin(hour_angle), np.cos(hour_angle)]

    return week_norm + hour_norm

def normalize_latlon(lat, lon):
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat_norm = [np.sin(lat_rad), np.cos(lat_rad)]
    lon_norm = [np.sin(lon_rad), np.cos(lon_rad)]
    return lat_norm + lon_norm

# Get center coordinates of the AOI
lat_center = (bbox[1] + bbox[3]) / 2
lon_center = (bbox[0] + bbox[2]) / 2

# Normalize time and location
current_date = pd.to_datetime(current_items[0].properties['datetime'])
two_weeks_date = pd.to_datetime(two_weeks_items[0].properties['datetime'])

current_time_norm = normalize_timestamp(current_date)
two_weeks_time_norm = normalize_timestamp(two_weeks_date)
latlon_norm = normalize_latlon(lat_center, lon_center)

# Convert to tensors
current_time_tensor = torch.tensor(current_time_norm, dtype=torch.float32).unsqueeze(0).to(device)
two_weeks_time_tensor = torch.tensor(two_weeks_time_norm, dtype=torch.float32).unsqueeze(0).to(device)
latlon_tensor = torch.tensor(latlon_norm, dtype=torch.float32).unsqueeze(0).to(device)

print("Current time norm:", current_time_norm)
print("Two weeks time norm:", two_weeks_time_norm)
print("Lat/Lon norm:", latlon_norm)

# Wavelengths for the bands (in nanometers)
# These values should match the model's expected input
waves = torch.tensor([492.4, 559.8, 664.6, 832.8], dtype=torch.float32).to(device)  # B02, B03, B04, B08

# Ground Sample Distance (GSD)
gsd = torch.tensor(10.0, dtype=torch.float32).to(device)
print("Waves:", waves)
print("GSD:", gsd)

def generate_embeddings(chips, time_tensor):
    embeddings = []
    with torch.no_grad():
        for i in range(len(chips)):
            pixels = chips[i].unsqueeze(0).to(device)  # Add batch dimension
            input_dict = {
                'pixels': pixels,
                'time': time_tensor,
                'latlon': latlon_tensor,
                'waves': waves,
                'gsd': gsd,
            }
            # Pass through the encoder
            unmsk_patch, _, _, _ = model.model.encoder(input_dict)
            # Extract class token embedding
            class_token_embedding = unmsk_patch[:, 0, :].cpu().numpy()
            embeddings.append(class_token_embedding)
    embeddings = np.vstack(embeddings)
    return embeddings

# Generate embeddings for current date
print("Generating embeddings for current date...")
current_embeddings = generate_embeddings(current_chips, current_time_tensor)

# Generate embeddings for two weeks ago
print("Generating embeddings for two weeks ago...")
two_weeks_embeddings = generate_embeddings(two_weeks_chips, two_weeks_time_tensor)

print("Current embeddings shape:", current_embeddings.shape)
print("Two weeks embeddings shape:", two_weeks_embeddings.shape)

# Check if the number of embeddings matches
assert current_embeddings.shape == two_weeks_embeddings.shape, "Embeddings shapes do not match!"

# Compute absolute differences
embedding_differences = np.abs(current_embeddings - two_weeks_embeddings)

# Compute change scores (e.g., L2 norm)
change_scores = np.linalg.norm(embedding_differences, axis=1)

print("Change scores statistics:")
print(f"Min: {change_scores.min()}")
print(f"Max: {change_scores.max()}")
print(f"Mean: {change_scores.mean()}")
print(f"Std: {change_scores.std()}")

# Set threshold (e.g., top 5% most changed areas)
threshold = np.percentile(change_scores, 95)

# Get indices of significant changes
significant_changes = change_scores > threshold

# Number of significant changes
print(f"Number of significant changes: {np.sum(significant_changes)}")
