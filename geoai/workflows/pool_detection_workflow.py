from langgraph.graph import Graph

from agents.query_agent import QueryAgent
from agents.data_retrieval_agent import DataRetrievalAgent
from agents.detection_agent import DetectionAgent
from agents.result_synthesis_agent import ResultSynthesisAgent

class PoolDetectionWorkflow:
    def __init__(self, query):
        self.query = query
        self.graph = Graph()
        self.query_agent = QueryAgent()
        self.data_agent = DataRetrievalAgent()
        self.detection_agent = DetectionAgent()
        self.result_agent = ResultSynthesisAgent()

    def build_workflow(self):
        query_node = self.graph.add_node(self.query_agent.interpret_query, name='QueryAgent')
        data_node = self.graph.add_node(self.data_agent.retrieve_data, name='DataRetrievalAgent')
        detection_node = self.graph.add_node(self.detection_agent.detect_pools, name='DetectionAgent')
        result_node = self.graph.add_node(self.result_agent.generate_results, name='ResultSynthesisAgent')

        self.graph.add_edge(query_node, data_node)
        self.graph.add_edge(data_node, detection_node)
        self.graph.add_edge(detection_node, result_node)

    def execute_workflow(self):
        self.build_workflow()
        results = self.graph.execute()
        map_filepath, report_filepath = results['ResultSynthesisAgent']
        return map_filepath, report_filepath

