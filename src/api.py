from contextlib import asynccontextmanager

import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import ThreatDetector


@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.detector = ThreatDetector()

    print("Threat detection API is ready.")

    yield

    print("Threat detection API shutting down.")


app = FastAPI(
    title="Cyber Threat Detection API",
    description=(
        "API for detecting malicious network traffic "
        "using Random Forest."
    ),
    version="1.0.0",
    lifespan=lifespan
)


class PredictionRequest(BaseModel):

    features: dict[str, float] = Field(
        ...,
        description="Network traffic features for prediction."
    )


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "Cyber Threat Detection API"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    detector = app.state.detector

    input_df = pd.DataFrame(
        [request.features]
    )

    try:

        result = detector.predict(input_df).iloc[0]

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    return {
        "prediction": int(result["prediction"]),
        "label": result["label"],
        "attack_probability": float(
            result["attack_probability"]
        )
    }
