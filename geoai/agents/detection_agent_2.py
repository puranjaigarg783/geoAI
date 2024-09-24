from models.pool_detector import PoolDetector
import geopandas as gpd
from shapely.geometry import Point
import random

class DetectionAgent:
    def __init__(self):
        self.detector = PoolDetector()

    def run(self, geodata):
        """
        Detects pools in the geospatial data.
        """
        # Simulate detection with random points
        num_pools = 100  # Simulated number of pools
        lats = [random.uniform(33.7, 34.3) for _ in range(num_pools)]
        lons = [random.uniform(-118.7, -117.9) for _ in range(num_pools)]
        points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
        pool_locations = gpd.GeoDataFrame({'geometry': points}, crs="EPSG:4326")
        return pool_locations

