from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from utils.config import OLLAMA_API_URL, OLLAMA_MODEL_NAME

class CodeGenerationAgent:
    def __init__(self):
        # Initialize the LLM using LangChain's Ollama interface
        self.llm = Ollama(base_url=OLLAMA_API_URL, model=OLLAMA_MODEL_NAME)
        # Define the prompt template using LangChain's PromptTemplate
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template="""
You are an assistant that generates Python code to perform geospatial analysis using Prithvi models.

User Query:
{query}

Constraints:
- The code should be self-contained and executable.
- Use appropriate libraries (e.g., geopandas, folium, transformers).
- Include comments for clarity.
- Ensure the code is safe and does not perform any malicious operations.
- When accessing Hugging Face models or datasets, include code to handle authentication tokens securely.
- The code should read the Hugging Face token from an environment variable named 'HUGGINGFACE_TOKEN' and use it when loading models or datasets.
- Do not hardcode any credentials or tokens in the code.

Generated Python Code:
"""
        )

    def generate_code(self, query):
        # Render the prompt with the user's query
        prompt = self.prompt_template.format(query=query)
        # Use the LLM to generate the code
        response = self.llm(prompt)
        return response.strip()

