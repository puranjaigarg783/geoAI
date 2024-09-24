# Import necessary libraries
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import os
from shapely.geometry import Point
import random

# Set random seed for reproducibility
random.seed(42)

# Create output directory
os.makedirs('output', exist_ok=True)

# Simulate pool locations in Los Angeles
num_pools = 100
lats = [random.uniform(33.7, 34.3) for _ in range(num_pools)]
lons = [random.uniform(-118.7, -117.9) for _ in range(num_pools)]
points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
pool_locations = gpd.GeoDataFrame({'geometry': points}, crs="EPSG:4326")

# Save pool locations to a file
pool_locations.to_file('output/pool_locations.geojson', driver='GeoJSON')

# Create a map with pool locations
m = folium.Map(location=[34.05, -118.25], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)
for idx, row in pool_locations.iterrows():
    folium.Marker(location=[row.geometry.y, row.geometry.x]).add_to(marker_cluster)

# Save the map
map_filepath = 'output/pools_map.html'
m.save(map_filepath)
print(f"Map saved to: {map_filepath}")

# Generate a summary report
total_pools = len(pool_locations)
report_content = f"Swimming Pools in Los Angeles\nTotal Pools Detected: {total_pools}\n"
report_filepath = 'output/summary_report.txt'
with open(report_filepath, 'w') as report_file:
    report_file.write(report_content)
print(f"Report saved to: {report_filepath}")

