from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ClinicalQueryRequest(BaseModel):
    query_text: str = Field(min_length=3)
    top_k_retrieval: int = Field(default=3, ge=1, le=10)
    similarity_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    guideline_filter: Optional[str] = None

class GuidelineSourcePassage(BaseModel):
    guideline_id: str
    title: str
    section: str
    content: str
    cosine_similarity: float
    evidence_level: Literal["Grade 1A", "Grade 1B", "Grade 2C", "Expert Consensus"]

class ClinicalRAGResponse(BaseModel):
    query: str
    synthesized_answer: str
    grounded_sources: List[GuidelineSourcePassage]
    faithfulness_score: float
    hallucination_risk: Literal["LOW", "MEDIUM", "HIGH"]
    disclaimer: str
