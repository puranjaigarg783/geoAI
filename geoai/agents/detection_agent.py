from langchain.agents import Agent
from models.pool_detector import PoolDetector
import geopandas as gpd
from shapely.geometry import Point

class DetectionAgent(Agent):
    def __init__(self):
        self.detector = PoolDetector()

    def detect_pools(self, geodata):
        """
        Detects pools in the provided geospatial data.
        """

        import random
        from utils.config import REGION_BOUNDARIES

        num_pools = 100  # Simulated number of pools
        lats = [random.uniform(REGION_BOUNDARIES['lat_min'], REGION_BOUNDARIES['lat_max']) for _ in range(num_pools)]
        lons = [random.uniform(REGION_BOUNDARIES['lon_min'], REGION_BOUNDARIES['lon_max']) for _ in range(num_pools)]
        points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
        pool_locations = gpd.GeoDataFrame({'geometry': points}, crs="EPSG:4326")
        return pool_locations

