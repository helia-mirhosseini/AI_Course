from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError, confloat
import joblib 
import pandas as pd
import numpy as np
import pathlib
from typing import Annotated

APP_DIR = pathlib.Path(__file__).parent
MODEL_PATH = "/Users/heliamirhosseini/Documents/Helia/AI_TA/chapter2/chapter1/California_Housing/best_model.joblib"

app = FastAPI(title= "California Housing API", version="1.0.0")

# static and template
app.mount("/static", StaticFiles(directory=str(APP_DIR/"static")), name="static")
template = Jinja2Templates(directory=str(APP_DIR/"templates"))

# loading the model
pipeline  = joblib.load(MODEL_PATH)

LOG_TARGET = False

class HouseFeatures(BaseModel): 
    MedInc:  Annotated[float, Field(ge = 0, default=4.0, description= "Median Income")]
    HouseAge: Annotated[float, Field(ge=0, default=25.0, description="Median age of the houses")]
    AveRooms: Annotated[float, Field(ge=0, default=5.5, description="Average number of rooms per household")]
    AveBedrms: Annotated[float, Field(ge=0, default=1.0, description="Average number of bedrooms per household")]
    Population: Annotated[float, Field(ge=0, default=1000.0, description="Total population in the district")]
    Latitude: float = Field(34.0)
    Longitude: float = Field(-118.0)

TRAIN_FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "Latitude", "Longitude"
]

def predict_from_features(feat: HouseFeatures) -> float:
    row = feat.model_dump()
    X = pd.DataFrame([row])

    # keep exactly the columns used in training, in the same order
    try:
        X = X[TRAIN_FEATURES].astype(float)
    except KeyError as e:
        raise ValueError(f"Missing or extra features. Expected exactly: {TRAIN_FEATURES}. Got: {list(X.columns)}") from e

    yhat = pipeline.predict(X)
    if LOG_TARGET:
        yhat = np.expm1(yhat)
    return float(np.asarray(yhat).ravel()[0])

@app.get("/", response_class=HTMLResponse)
def form_get(request:Request): 
    return template.TemplateResponse("index.html", {"request":request, "result": None})

@app.post("/", response_class=HTMLResponse)
def form_post(
    request: Request,
    MedInc: float = Form(...),
    HouseAge: float = Form(...),
    AveRooms: float = Form(...),
    AveBedrms: float = Form(...),
    Population: float = Form(...),
    Latitude: float = Form(...),
    Longitude: float = Form(...)
):
    try: 
        feats = HouseFeatures(
            MedInc=MedInc, HouseAge=HouseAge, AveRooms=AveRooms, AveBedrms=AveBedrms,
            Population=Population, Latitude=Latitude, Longitude=Longitude
        )

        pred = predict_from_features(feats)
        return template.TemplateResponse("result.html", {"request": request, "pred": pred, "feats": feats})
    except ValidationError as e :
        return template.TemplateResponse("index.html", {"request":request, "errors": e.errors()})
    
from fastapi.responses import JSONResponse
import traceback

@app.post("/predict")
def predict_api(item: HouseFeatures):
    try:
        pred = predict_from_features(item)
        return {"prediction": pred}
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "traceback": traceback.format_exc()},
        )

@app.get("/health")
def health():
    return {"status": "ok"}


