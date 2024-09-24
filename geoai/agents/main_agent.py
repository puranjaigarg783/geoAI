from langchain.agents import AgentExecutor, Tool
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from utils.config import OLLAMA_API_URL
from langgraph.graph import Graph
from agents.data_agent import DataAgent
from agents.detection_agent import DetectionAgent
from agents.synthesis_agent import SynthesisAgent

class MainAgent:
    def __init__(self):
        self.llm = Ollama(base_url=OLLAMA_API_URL, model='your-llm-model-name')
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template="You are an AI assistant that interprets the following user query and decides which tools to use: {query}"
        )
        self.tools = [
            Tool(name="DataAgent", func=DataAgent().run, description="Retrieves geospatial data"),
            Tool(name="DetectionAgent", func=DetectionAgent().run, description="Detects pools in geospatial data"),
            Tool(name="SynthesisAgent", func=SynthesisAgent().run, description="Generates map and report")
        ]
        self.agent = AgentExecutor.from_agent_and_tools(
            agent=self.llm,
            tools=self.tools,
            verbose=True
        )
        self.graph = Graph()

    def run(self, query):
        data_node = self.graph.add_node(DataAgent().run, name='DataAgent')
        detection_node = self.graph.add_node(DetectionAgent().run, name='DetectionAgent')
        synthesis_node = self.graph.add_node(SynthesisAgent().run, name='SynthesisAgent')

        self.graph.add_edge(data_node, detection_node)
        self.graph.add_edge(detection_node, synthesis_node)

        result = self.graph.execute()
        return result

