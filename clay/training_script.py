# training_script.py

# Import necessary modules
import sys
sys.path.append("../..")  # Adjust the path to your project's root if necessary

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
from rasterio.enums import Resampling
from sklearn import svm
from torchvision.transforms import v2
import joblib  # For saving the trained classifier

from src.model import ClayMAEModule

# Set the new location and date range of interest for training data
lat, lon = 38.5800, -120.4500
start = "2021-08-10"
end = "2021-09-25"

# Get data from STAC catalog
STAC_API = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# Search the catalog
catalog = pystac_client.Client.open(STAC_API)
search = catalog.search(
    collections=[COLLECTION],
    datetime=f"{start}/{end}",
    bbox=(lon - 0.00001, lat - 0.00001, lon + 0.00001, lat + 0.00001),
    max_items=100,
    query={"eo:cloud_cover": {"lt": 80}},
)

all_items = search.get_all_items()

# Reduce to one item per date to avoid duplicates
items = []
dates = []
for item in all_items:
    if item.datetime.date() not in dates:
        items.append(item)
        dates.append(item.datetime.date())

print(f"Found {len(items)} items for training.")

# Create a bounding box around the point of interest
epsg = items[0].properties["proj:epsg"]
poidf = gpd.GeoDataFrame(
    pd.DataFrame(),
    crs="EPSG:4326",
    geometry=[Point(lon, lat)],
).to_crs(epsg)
coords = poidf.iloc[0].geometry.coords[0]

# Define the size and resolution
size = 256
gsd = 10  # Ground Sample Distance in meters
bounds = (
    coords[0] - (size * gsd) // 2,
    coords[1] - (size * gsd) // 2,
    coords[0] + (size * gsd) // 2,
    coords[1] + (size * gsd) // 2,
)

# Retrieve the imagery data
stack = stackstac.stack(
    items,
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

# Load the Clay model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ckpt = "https://clay-model-ckpt.s3.amazonaws.com/v0.5.7/mae_v0.5.7_epoch-13_val-loss-0.3098.ckpt"

model = ClayMAEModule.load_from_checkpoint(
    ckpt, metadata_path="../../configs/metadata.yaml", shuffle=False, mask_ratio=0
)
model.eval()
model = model.to(device)

# Prepare band metadata
platform = "sentinel-2-l2a"
metadata = Box(yaml.safe_load(open("../../configs/metadata.yaml")))
mean = []
std = []
waves = []
for band in stack.band.values:
    mean.append(metadata[platform].bands.mean[str(band)])
    std.append(metadata[platform].bands.std[str(band)])
    waves.append(metadata[platform].bands.wavelength[str(band)])

transform = v2.Compose([v2.Normalize(mean=mean, std=std)])

# Normalize timestamps and lat/lon
def normalize_timestamp(date):
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24
    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

def normalize_latlon(lat, lon):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))

datetimes = stack.time.values.astype("datetime64[s]").tolist()
times = [normalize_timestamp(dat) for dat in datetimes]
week_norm = [t[0] for t in times]
hour_norm = [t[1] for t in times]

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

# Generate embeddings using the Clay model
with torch.no_grad():
    unmsk_patch, _, _, _ = model.model.encoder(datacube)

embeddings = unmsk_patch[:, 0, :].cpu().numpy()
print(f"Embeddings shape: {embeddings.shape}")

# Label the images (0: Cloud, 1: Forest, 2: Fire)
labels = np.array([1, 1, 0, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

# Split into training and testing sets
num_samples = len(embeddings)
num_train = int(0.7 * num_samples)  # Use 70% for training
fit_indices = list(range(num_train))
test_indices = list(range(num_train, num_samples))

# Train the Support Vector Machine classifier
clf = svm.SVC()
clf.fit(embeddings[fit_indices] + 100, labels[fit_indices])

# Evaluate the classifier
predictions = clf.predict(embeddings[test_indices] + 100)
accuracy = np.mean(predictions == labels[test_indices])
print(f"Classifier accuracy on test set: {accuracy * 100:.2f}%")

# Save the trained classifier to disk
joblib.dump(clf, 'svm_classifier.joblib')
print("Classifier trained and saved to 'svm_classifier.joblib'.")

# Print detailed results
print("\nDetailed classification results:")
for i, (true_label, pred_label) in enumerate(zip(labels[test_indices], predictions)):
    print(f"Image {test_indices[i]}: True label: {true_label}, Predicted: {pred_label}")