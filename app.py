from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from project.entity.config import Model_Trainer_Config
from project.exception import CustomException
from project.logger import logging
from project.utils import load_object, read_yaml
from project.utils.feature_engineering import empty_string_columns
from project.constants import COLUMN_YAML_FILE_PATH

app = FastAPI(title="Telco Churn Prediction API")
templates = Jinja2Templates(directory="project/templates")

model = None
column_schema = read_yaml(COLUMN_YAML_FILE_PATH)


def _load_prediction_model():
    """
    Load prediction pipeline model with fallback paths.
    """
    global model

    if model is not None:
        return model

    cfg = Model_Trainer_Config()
    candidate_paths = [
        cfg.final_model_path,
        "final_model/prediction_model/pred_model.pkl",
    ]

    last_error = None
    for path in candidate_paths:
        try:
            model = load_object(path)
            logging.info(f"Prediction model loaded from: {path}")
            return model
        except Exception as e:
            last_error = e

    raise CustomException(f"Unable to load prediction model. Last error: {last_error}", __import__("sys"))


def _apply_binary_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    binary_cols = column_schema.get("binary_categorical_columns", {})
    for col, details in binary_cols.items():
        if col in df_copy.columns and col != "Churn":
            mapping = details.get("mapping", {})
            df_copy[col] = df_copy[col].map(mapping)

    return df_copy


def _apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    log_cols = column_schema.get("log_transform_col", [])
    for col in log_cols:
        if col in df_copy.columns:
            df_copy[col] = np.log1p(pd.to_numeric(df_copy[col], errors="coerce").clip(lower=0))
    return df_copy


def _preprocess_payload(payload: dict) -> dict:
    """
    Apply the same input cleaning used in model evaluation before prediction.
    """
    df = pd.DataFrame([payload])
    df = empty_string_columns(df)
    df = _apply_binary_mapping(df)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        if "MonthlyCharges" in df.columns:
            df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
            df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
        else:
            df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = _apply_log_transform(df)

    drop_cols = column_schema.get("drop_columns", [])
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    return df.iloc[0].to_dict()


@app.on_event("startup")
def startup_event():
    _load_prediction_model()


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "telco-churn-api"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    gender: str = Form(...),
    SeniorCitizen: int = Form(...),
    Partner: str = Form(...),
    Dependents: str = Form(...),
    tenure: int = Form(...),
    PhoneService: str = Form(...),
    MultipleLines: str = Form(...),
    InternetService: str = Form(...),
    OnlineSecurity: str = Form(...),
    OnlineBackup: str = Form(...),
    DeviceProtection: str = Form(...),
    TechSupport: str = Form(...),
    StreamingTV: str = Form(...),
    StreamingMovies: str = Form(...),
    Contract: str = Form(...),
    PaperlessBilling: str = Form(...),
    PaymentMethod: str = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: Optional[float] = Form(None),
):
    try:
        model_obj = _load_prediction_model()

        if TotalCharges is None:
            TotalCharges = float(MonthlyCharges)

        input_data = {
            "gender": gender,
            "SeniorCitizen": int(SeniorCitizen),
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": int(tenure),
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": float(MonthlyCharges),
            "TotalCharges": float(TotalCharges),
        }
        input_data = _preprocess_payload(input_data)

        prediction_df = model_obj.predict(input_data)
        prediction_value = int(prediction_df["prediction"].iloc[0])
        result = "Customer is likely to churn" if prediction_value == 1 else "Customer is likely to stay"

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": result, "prediction_value": prediction_value},
        )

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": f"Prediction failed: {e}"},
            status_code=400,
        )


@app.post("/predict-json")
async def predict_json(payload: dict):
    try:
        model_obj = _load_prediction_model()
        processed_payload = _preprocess_payload(payload)
        prediction_df = model_obj.predict(processed_payload.to_numpy())
        prediction_value = int(prediction_df["prediction"].iloc[0])
        return JSONResponse(
            {
                "prediction": prediction_value,
                "label": "churn" if prediction_value == 1 else "no_churn",
            }
        )
    except Exception as e:
        logging.error(f"JSON prediction failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

## uvicorn app:app --reload
