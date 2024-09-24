import os
import requests
import geopandas as gpd
from shapely.geometry import box
from .config import DATA_DIR, REGION_BOUNDARIES

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def get_geospatial_data():
    """
    Retrieves geospatial data for the specified region.
    """
    ensure_data_dir()
    
    bbox = box(REGION_BOUNDARIES['lon_min'], REGION_BOUNDARIES['lat_min'],
               REGION_BOUNDARIES['lon_max'], REGION_BOUNDARIES['lat_max'])
    gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs="EPSG:4326")
    return gdf

def save_data(gdf, filename):
    """
    Saves the GeoDataFrame to a file.
    """
    filepath = os.path.join(DATA_DIR, filename)
    gdf.to_file(filepath, driver='GeoJSON')

