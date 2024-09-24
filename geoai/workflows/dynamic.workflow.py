from langgraph.graph import Graph
from agents.code_generation_agent import CodeGenerationAgent
from agents.execution_agent import ExecutionAgent
from agents.synthesis_agent import SynthesisAgent

class DynamicWorkflow:
    def __init__(self, query):
        self.query = query
        self.graph = Graph()
        self.code_gen_agent = CodeGenerationAgent()
        self.exec_agent = ExecutionAgent()
        self.synthesis_agent = SynthesisAgent()

    def build_workflow(self):
        
        code_gen_node = self.graph.add_node(
            lambda: self.code_gen_agent.generate_code(self.query), name='CodeGeneration'
        )
        execution_node = self.graph.add_node(
            self.exec_agent.execute, name='Execution'
        )
        synthesis_node = self.graph.add_node(
            self.synthesis_agent.synthesize, name='Synthesis'
        )

        
        self.graph.add_edge(code_gen_node, execution_node)
        self.graph.add_edge(execution_node, synthesis_node)

    def execute_workflow(self):
        self.build_workflow()
        
        results = self.graph.execute()
        final_result = results['Synthesis']
        return final_result

