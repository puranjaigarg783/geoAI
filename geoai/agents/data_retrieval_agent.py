from langchain.agents import Agent
from utils.data_utils import get_geospatial_data

class DataRetrievalAgent(Agent):
    def __init__(self):
        pass

    def retrieve_data(self):
        """
        Retrieves geospatial data needed for analysis.
        """
        geodata = get_geospatial_data()
        return geodata

