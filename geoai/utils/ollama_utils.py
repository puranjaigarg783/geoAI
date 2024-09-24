import requests
from utils.config import OLLAMA_API_URL, OLLAMA_MODEL_NAME, OLLAMA_API_KEY

def ollama_generate(prompt):
    """
    Sends a prompt to the Ollama API and returns the generated response.
    """
    headers = {}
    if OLLAMA_API_KEY:
        
        headers['Authorization'] = f"Bearer {OLLAMA_API_KEY}"

    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt
    }

    
    response = requests.post(f"{OLLAMA_API_URL}/api/generate", json=payload, headers=headers)
    
    
    response.raise_for_status()
    
    
    result = response.json()
    return result['response']

