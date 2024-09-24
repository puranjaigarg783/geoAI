from utils.ollama_utils import ollama_generate

class CodeGenerationAgent:
    def __init__(self):
        pass

    def generate_code(self, query):
        """
        Uses the LLM to generate code based on the user's query.
        """
        prompt = f"""
You are an assistant that generates Python code to perform geospatial analysis using Prithvi models.

User Query:
{query}

Constraints:
- The code should be self-contained and executable.
- Use appropriate libraries (e.g., geopandas, folium, transformers).
- Include comments for clarity.
- Ensure the code is safe and does not perform any malicious operations.

Generated Python Code:
"""
        code = ollama_generate(prompt)
        return code.strip()

