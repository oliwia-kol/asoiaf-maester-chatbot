# src/config.py
"""
Application Configuration Management

This module demonstrates enterprise-grade configuration using Pydantic.
I separate configs by domain (database, models, processing) for maintainability.

Key Concepts:
- Type validation prevents runtime errors
- Environment variables override defaults
- Hierarchical structure scales with application growth
- Validation rules catch misconfigurations early
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, validator, Field
from pathlib import Path


class ProcessingConfig(BaseSettings):
    """
    Text processing configuration.
    
    Architecture Decision: Keep processing settings separate from other concerns
    so data scientists can modify these without affecting API or database settings.
    """
    # Chunking parameters - these directly impact RAG quality
    chunk_size: int = Field(
        default=500, 
        env="CHUNK_SIZE",
        description="Target words per chunk (balance context vs precision)"
    )
    chunk_overlap: int = Field(
        default=50, 
        env="CHUNK_OVERLAP",
        description="Overlapping words between chunks (prevents boundary loss)"
    )
    min_chunk_size: int = Field(
        default=100, 
        env="MIN_CHUNK_SIZE",
        description="Minimum chunk size to avoid tiny fragments"
    )
    
    # Directory structure
    data_dir: str = Field(default="data", env="DATA_DIR")
    raw_data_dir: str = Field(default="data/raw", env="RAW_DATA_DIR")  
    processed_data_dir: str = Field(default="data/processed", env="PROCESSED_DATA_DIR")
    
    # Processing toggles
    clean_text: bool = Field(default=True, env="CLEAN_TEXT")
    extract_metadata: bool = Field(default=True, env="EXTRACT_METADATA")
    
    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        """Business Rule: Overlap cannot exceed chunk size"""
        chunk_size = values.get('chunk_size', 500)
        if v >= chunk_size:
            raise ValueError(f'Chunk overlap ({v}) must be smaller than chunk size ({chunk_size})')
        return v
    
    @validator('data_dir', 'raw_data_dir', 'processed_data_dir')
    def ensure_directories_exist(cls, v):
        """Infrastructure: Create directories if they don't exist"""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class ModelConfig(BaseSettings):
    """
    AI Model configuration.
    
    Architecture Decision: Separate model configs enable easy A/B testing
    and model swapping without code changes.
    """
    # Model selection - these affect performance and quality
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL_NAME",
        description="HuggingFace model for text embeddings"
    )
    
    # Generation parameters - control AI behavior
    max_tokens: int = Field(default=512, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    # Performance settings
    model_cache_dir: str = Field(default="./models", env="MODEL_CACHE_DIR")
    device: str = Field(default="cpu", env="DEVICE")  # or "cuda" for GPU
    
    @validator('temperature')
    def validate_temperature(cls, v):
        """AI Parameter Validation: Temperature affects randomness"""
        if not 0 <= v <= 2:
            raise ValueError('Temperature must be between 0 (deterministic) and 2 (very random)')
        return v

class DatabaseConfig(BaseSettings):
    """
    Vector database configuration.
    
    Architecture Decision: Abstract database details so we can swap
    ChromaDB for Pinecone/Weaviate later without changing business logic.
    """
    # ChromaDB settings
    persist_directory: str = Field(
        default="./data/embeddings", 
        env="CHROMA_PERSIST_DIRECTORY"
    )
    collection_name: str = Field(
        default="asoiaf_knowledge", 
        env="COLLECTION_NAME"
    )
    
    # Vector settings
    vector_dimension: int = Field(default=384, env="VECTOR_DIMENSION")
    max_elements: int = Field(default=10000, env="MAX_ELEMENTS")
    
    @validator('persist_directory')
    def ensure_db_directory(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

class APIConfig(BaseSettings):
    """
    FastAPI server configuration.
    """
    host: str = Field(default="0.0.0.0", env="FASTAPI_HOST")
    port: int = Field(default=8000, env="FASTAPI_PORT")
    debug: bool = Field(default=False, env="DEBUG_MODE")
    
    # Request limits - prevent abuse
    max_query_length: int = Field(default=1000, env="MAX_QUERY_LENGTH")
    timeout_seconds: int = Field(default=30, env="TIMEOUT_SECONDS")


class Settings(BaseSettings):
    """
    Main application settings combining all configuration domains.
    
    Architecture Pattern: Composition over inheritance
    - Each domain is a separate object
    - Easy to mock individual sections for testing
    - Clear ownership of configuration sections
    """
    
    # Configuration sections
    processing: ProcessingConfig = ProcessingConfig()
    model: ModelConfig = ModelConfig()
    database: DatabaseConfig = DatabaseConfig()
    api: APIConfig = APIConfig()

# Global application settings
    app_name: str = Field(default="ASOIAF Maester Chatbot", env="APP_NAME")
    version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def is_production(self) -> bool:
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        return self.environment.lower() == "development"


# Global settings instance - Singleton pattern
settings = Settings()

def get_settings() -> Settings:
    """
    Dependency injection function for FastAPI.
    
    This pattern allows easy testing and configuration swapping.
    """
    return settings


# Example usage and validation
if __name__ == "__main__":
    print("🏰 ASOIAF Configuration Loaded")
    print(f"Environment: {settings.environment}")
    print(f"Chunk Size: {settings.processing.chunk_size}")
    print(f"Model: {settings.model.embedding_model_name}")
    print(f"Database: {settings.database.persist_directory}")