import os
from dotenv import load_dotenv

load_dotenv()
OLLAMA_API_URL = 'http://localhost:11434'  # Replace with your Ollama server URL
OLLAMA_MODEL_NAME = 'your-ollama-model-name'  # Replace with your Ollama model name
OLLAMA_API_KEY = os.getenv('OLLAMA_API_KEY')

PRITHVI_MODEL_NAME = 'ibm-nasa-geospatial/Prithvi-100M'


OUTPUT_DIR = 'output/'


EXECUTION_TIMEOUT = 30  # Timeout in seconds for code execution


MAP_CENTER = [34.05, -118.25]  # Los Angeles coordinates
MAP_ZOOM_START = 10


SEED = 42

