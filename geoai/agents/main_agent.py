from langchain.agents import initialize_agent, Tool
from langchain.llms import Ollama
from utils.config import OLLAMA_API_URL, OLLAMA_MODEL_NAME
from agents.code_generation_agent import CodeGenerationAgent
from agents.execution_agent import ExecutionAgent
from agents.synthesis_agent import SynthesisAgent

class MainAgent:
    def __init__(self):
        self.llm = Ollama(base_url=OLLAMA_API_URL, model=OLLAMA_MODEL_NAME)
        self.code_gen_agent = CodeGenerationAgent()
        self.execution_agent = ExecutionAgent()
        self.synthesis_agent = SynthesisAgent()
        self.tools = [
            Tool(
                name="CodeGenerator",
                func=self.code_gen_agent.generate_code,
                description="Generates code based on user query."
            ),
            Tool(
                name="CodeExecutor",
                func=self.execution_agent.execute,
                description="Executes generated code securely."
            ),
            Tool(
                name="ResultSynthesizer",
                func=self.synthesis_agent.synthesize,
                description="Summarizes execution output."
            ),
        ]
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="zero-shot-react-description",
            verbose=True
        )

    def run(self, query):
        result = self.agent.run(query)
        return result

