import requests
from utils.config import OLLAMA_API_URL, OLLAMA_MODEL_NAME

def ollama_generate(prompt):
    """
    Sends a prompt to the Ollama API and returns the generated response.
    """
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt
    }
    response = requests.post(f"{OLLAMA_API_URL}/api/generate", json=payload)
    response.raise_for_status()
    result = response.json()
    return result['response']

