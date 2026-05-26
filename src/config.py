import os
from typing import Literal
import yaml
from pydantic import BaseModel, Field, ValidationError

class Settings(BaseModel):
    chunk_size: int = Field(..., description="Maximum size of document chunk", gt=0)
    chunk_overlap: int = Field(..., description="Overlap size between chunks", ge=0)
    persist_directory: str = Field(..., description="Directory path for vector store persistence")
    embedding_model_name: str = Field(..., description="Name of local or API embedding model")
    llm_provider: Literal["ollama", "openai"] = Field(..., description="Inference provider")
    llm_model: str = Field(..., description="LLM model identifier")
    llm_temperature: float = Field(0.0, description="Temperature for inference generation", ge=0.0, le=1.0)

def load_settings(config_path: str = None) -> Settings:
    """Loads, parses, and validates configurations from a YAML file.
    
    Args:
        config_path: Optional path to the configuration yaml. If not provided,
                     defaults to 'config.yaml' at the project root.
    
    Returns:
        An instance of Settings representing validated configuration properties.
        
    Raises:
        FileNotFoundError: If the config file path does not exist.
        ValidationError: If the YAML contents violate the type boundaries of Settings.
    """
    if config_path is None:
        # Default path relative to the root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(root_dir, "config.yaml")
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}
        
    try:
        return Settings(**config_data)
    except ValidationError as e:
        # Explicit loud logging/raising for production stability
        print(f"Configuration validation failed: {e}")
        raise e
