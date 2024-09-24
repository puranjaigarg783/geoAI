from langchain.agents import Agent
from utils.visualization_utils import create_map, generate_report

class ResultSynthesisAgent(Agent):
    def __init__(self):
        pass

    def generate_results(self, pool_locations):
        """
        Generates the map and report based on detected pools.
        """
        map_filepath = create_map(pool_locations)
        report_filepath = generate_report(pool_locations)
        return map_filepath, report_filepath

