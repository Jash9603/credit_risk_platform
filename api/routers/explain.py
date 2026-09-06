from fastapi import APIRouter, HTTPException

from api.schemas import ExplainResponse, ExplanationItem, PredictRequest
from src.explainability.shap_explainer import explain

router = APIRouter(tags=["explain"])


@router.post("/explain", response_model=ExplainResponse)
def explain_applicant(request: PredictRequest):
    try:
        items = explain(request.sk_id_curr)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ExplainResponse(
        sk_id_curr=request.sk_id_curr,
        explanations=[ExplanationItem(**item) for item in items],
    )
