from langchain.llms import Ollama
from utils.config import OLLAMA_API_URL, OLLAMA_MODEL_NAME
from utils.ollama_utils import ollama_generate

class CodeGenerationAgent:
    def __init__(self):
        self.llm = Ollama(base_url=OLLAMA_API_URL, model=OLLAMA_MODEL_NAME)

    def generate_code(self, query):
        """
        Uses the LLM to generate code based on the user's query.
        """
        prompt = f"""
You are an AI assistant that generates Python code to perform geospatial analysis using Prithvi or Clay models based on the user's query.

User Query:
{query}

Constraints:
- The code should be self-contained and executable.
- Use appropriate libraries and handle imports.
- Ensure the code is safe and does not perform any malicious operations.

Generated Python Code (include comments for clarity):
"""
        code = ollama_generate(prompt)
        return code.strip()

