from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.inference.detector import AnomalyDetector

router = APIRouter()
detector = AnomalyDetector()

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image string

class ImageResponse(BaseModel):
    prediction: str
    confidence: float

@router.post("/detect", response_model=ImageResponse)
async def detect_anomaly(request: ImageRequest):
    try:
        prediction, confidence = detector.detect(request.image)
        return ImageResponse(prediction=prediction, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))