"""Pydantic request/response models for the API."""
from typing import Optional

from pydantic import BaseModel


class PredictRequest(BaseModel):
    sk_id_curr: int


class PredictResponse(BaseModel):
    sk_id_curr: int
    default_probability: float
    risk_score: int
    risk_band: str
    actual_target: Optional[int] = None


class ExplanationItem(BaseModel):
    feature: str
    value: str
    shap_value: float
    sentence: str


class ExplainResponse(BaseModel):
    sk_id_curr: int
    explanations: list[ExplanationItem]


class RuleCondition(BaseModel):
    feature: str
    op: str
    threshold: float


class Rule(BaseModel):
    conditions: list[RuleCondition]
    support: int
    observed_default_rate: float
    lift_vs_average: Optional[float] = None
    readable: str


class RulesResponse(BaseModel):
    rules: list[Rule]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    question: str
    standalone_question: str
    sql: Optional[str] = None
    rows: list[dict] = []
    answer: str
    refused: bool


class EDAInsight(BaseModel):
    title: str
    description: str
    chart_url: str


class EDASummaryResponse(BaseModel):
    n_applicants: int
    n_features: int
    default_rate_pct: float
    n_columns_with_missing: int
