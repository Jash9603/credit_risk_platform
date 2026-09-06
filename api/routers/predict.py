from fastapi import APIRouter, HTTPException

from api.schemas import PredictRequest, PredictResponse
from src.ml.predict import predict

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict_applicant(request: PredictRequest):
    try:
        result = predict(request.sk_id_curr)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return PredictResponse(**result)
