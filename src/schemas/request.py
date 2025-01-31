from typing import List, Optional
from pydantic import BaseModel, HttpUrl


class PredictionRequest(BaseModel):
    query: str
    id: int


class ResponseSchema(BaseModel):
    answer: Optional[int] = None
    reasoning: str

    class Config:
        extra = "forbid"


class PredictionResponse(BaseModel):
    id: int
    answer: Optional[int] = None
    reasoning: str
    sources: List[str] = []

    def model_dump(self, **kwargs):
        # Convert HttpUrl objects to strings if present
        data = super().model_dump(**kwargs)
        if "sources" in data:
            data["sources"] = [str(url) for url in data["sources"]]
        return data
