import httpx
from openai import OpenAI
from src.config import load_settings

def generate_answer(prompt: str) -> str:
    """Sends the compiled prompt to the configured LLM provider and returns the answer.
    
    Args:
        prompt: The complete prompt text containing context directives and query.
        
    Returns:
        The generated natural language answer string.
        
    Raises:
        ValueError: If the llm_provider config is unsupported.
        ConnectionError: If communication with the local Ollama server fails.
    """
    settings = load_settings()
    
    if settings.llm_provider == "openai":
        if settings.openai_base_url:
            # For local compatible endpoints (like LM Studio), use custom base_url and bypass api_key checks
            client = OpenAI(base_url=settings.openai_base_url, api_key="lm-studio")
        else:
            client = OpenAI()
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.llm_temperature
        )
        return response.choices[0].message.content.strip()
        
    elif settings.llm_provider == "ollama":
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": settings.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.llm_temperature
            }
        }
        
        try:
            response = httpx.post(url, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            print(f"Error during Ollama API request: {e}")
            raise ConnectionError(f"Failed to connect to local Ollama server at {url}: {e}")
            
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
