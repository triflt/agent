from pydantic import BaseModel
from typing import Optional, List

class AssistantResponse(BaseModel):
    """Structured response from the assistant"""
    answer: Optional[int] = None
    reasoning: str

class ContextInfo(BaseModel):
    """Information about the context used"""
    text: str
    source_url: str

class PredictionResponse(BaseModel):
    """Final API response format"""
    id: int
    answer: Optional[int] = None
    reasoning: str
    sources: List[str] = [] 