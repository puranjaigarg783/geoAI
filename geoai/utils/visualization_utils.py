import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
from .config import MAP_CENTER, MAP_ZOOM_START, OUTPUT_DIR
import os

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def create_map(pool_locations):
    """
    Creates an interactive map with pool locations.
    """
    ensure_output_dir()
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM_START)
    marker_cluster = MarkerCluster().add_to(m)

    for idx, row in pool_locations.iterrows():
        lat = row.geometry.y
        lon = row.geometry.x
        folium.Marker(location=[lat, lon]).add_to(marker_cluster)

    map_filepath = os.path.join(OUTPUT_DIR, 'pools_map.html')
    m.save(map_filepath)
    return map_filepath

def generate_report(pool_locations):
    """
    Generates a simple text report about the detected pools.
    """
    ensure_output_dir()
    total_pools = len(pool_locations)
    report = f"Swimming Pools Detection Report\n\nTotal Pools Detected: {total_pools}\n"
    report_filepath = os.path.join(OUTPUT_DIR, 'report.txt')
    with open(report_filepath, 'w') as f:
        f.write(report)
    return report_filepath

