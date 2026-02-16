import sys
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from project.entity.config import Data_Transformation_Config, Model_Evaluation_Config
from project.entity.artifacts import (
    Data_Ingestion_Artifact,
    Model_Evaluation_Artifact,
    Model_Trainer_Artifact,
)
from project.entity.estimator import ChurnTargetMapping
from project.exception import CustomException
from project.logger import logging
from project.utils import load_object, read_yaml
from project.utils.feature_engineering import collapse_redundant_columns, empty_string_columns
from project.constants import COLUMN_YAML_FILE_PATH, Target_Column


@dataclass
class Evaluate_Model_Response:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float


class Model_Evaluation:
    def __init__(
        self,
        model_eval_config: Model_Evaluation_Config,
        data_ingestion_artifact: Data_Ingestion_Artifact,
        model_trainer_artifact: Model_Trainer_Artifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._column_schema = read_yaml(COLUMN_YAML_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_best_model(self, trained_model_path: Optional[str] = None):
        """
        Load local baseline model (AWS mode intentionally disabled).
        Set `model_eval_config.s3_model_key_path` to a local model path.
        """
        try:
            local_model_path = self.model_eval_config.s3_model_key_path
            trained_abs = os.path.abspath(trained_model_path) if trained_model_path else None
            basename = os.path.basename(local_model_path) if local_model_path else None

            candidate_paths = []
            if local_model_path:
                candidate_paths.append(local_model_path)
            if basename:
                candidate_paths.extend(
                    [
                        os.path.join("artifact", "model_trainer", basename),
                        os.path.join("final_model", "prediction_model", basename),
                    ]
                )

            seen = set()
            unique_paths = []
            for path in candidate_paths:
                if path not in seen:
                    seen.add(path)
                    unique_paths.append(path)

            for path in unique_paths:
                path_abs = os.path.abspath(path)
                if trained_abs and path_abs == trained_abs:
                    continue
                if os.path.exists(path):
                    logging.info(f"Using local baseline model: {path}")
                    return load_object(file_path=path), path

            logging.info(
                f"Local baseline model not found at '{local_model_path}'. "
                "Evaluation will accept trained model by default."
            )


            # AWS production mode (enable later):
            # from project.entity.s3_estimator import AWSEstimator
            # bucket_name = self.model_eval_config.bucket_name
            # model_key = self.model_eval_config.s3_model_key_path
            # model_estimator = AWSEstimator(bucket_name=bucket_name, model_key=model_key)
            # if model_estimator.is_model_present():
            #     return model_estimator
            # return None
            
            return None, None
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _prepare_target(y: pd.Series) -> np.ndarray:
        """
        Convert labels to numeric 0/1 when possible.
        """
        if pd.api.types.is_numeric_dtype(y):
            return y.to_numpy()

        mapping = {
            "yes": 1,
            "no": 0,
            "churn": ChurnTargetMapping.churn,
            "no_churn": ChurnTargetMapping.no_churn,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        }

        normalized = y.astype(str).str.strip().str.lower().map(mapping)
        if normalized.isna().any():
            logging.warning("Target mapping contains unknown labels; using raw labels for evaluation.")
            return y.to_numpy()
        return normalized.to_numpy()

    @staticmethod
    def _to_1d_predictions(y_pred) -> np.ndarray:
        """
        Normalize model output to a 1D array for sklearn metrics.
        """
        if isinstance(y_pred, pd.DataFrame):
            if "prediction" in y_pred.columns:
                return y_pred["prediction"].to_numpy()
            return y_pred.iloc[:, 0].to_numpy()
        if isinstance(y_pred, pd.Series):
            return y_pred.to_numpy()
        if isinstance(y_pred, np.ndarray):
            return y_pred.ravel()
        return np.asarray(y_pred).ravel()

    def apply_binary_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply binary mapping from schema (Yes/No, Male/Female, etc.).
        """
        df_copy = df.copy()
        binary_cols = self._column_schema.get("binary_categorical_columns", {})

        for col, details in binary_cols.items():
            if col in df_copy.columns:
                mapping = details.get("mapping", {})
                df_copy[col] = df_copy[col].replace(mapping)

        return df_copy

    def apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log1p transform for configured columns.
        """
        df_copy = df.copy()
        log_cols = self._column_schema.get("log_transform_col", [])

        for col in log_cols:
            if col in df_copy.columns:
                df_copy[col] = np.log1p(df_copy[col].clip(lower=0))

        return df_copy

    def _transform_for_raw_model(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Build transformed features for raw sklearn model inference.
        """
        transform_cfg = Data_Transformation_Config()
        preprocessor = load_object(file_path=transform_cfg.transform_object_path)

        x_trans = preprocessor.transform(x)
        feature_names = preprocessor.get_feature_names_out()
        x_trans = pd.DataFrame(x_trans, columns=feature_names, index=x.index)
        x_trans = collapse_redundant_columns(x_trans)
        # Raw sklearn models in this project are trained from numpy arrays.
        # Return numpy here to keep fit/predict input style consistent.
        return x_trans.to_numpy()

    def _prepare_features_and_target(self) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare evaluation dataframe using transformation-aligned cleaning.
        """
        test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
        test_df = empty_string_columns(test_df)
        test_df = self.apply_binary_mapping(test_df)

        if "TotalCharges" in test_df.columns:
            test_df["TotalCharges"] = pd.to_numeric(test_df["TotalCharges"], errors="coerce")
            if "MonthlyCharges" in test_df.columns:
                test_df["TotalCharges"] = test_df["TotalCharges"].fillna(test_df["MonthlyCharges"])
            else:
                test_df["TotalCharges"] = test_df["TotalCharges"].fillna(0)

        test_df = self.apply_log_transform(test_df)

        drop_cols = self._column_schema.get("drop_columns", [])
        test_df.drop(columns=drop_cols, inplace=True, errors="ignore")

        x = test_df.drop(Target_Column, axis=1)
        y_true = self._prepare_target(test_df[Target_Column])
        return x, y_true

    def _predict_with_any_model(self, model_obj: Any, x_raw: pd.DataFrame) -> np.ndarray:
        """
        Predict robustly for both:
        - ProjectModel (raw features)
        - raw sklearn model (transformed features)
        """
        if hasattr(model_obj, "transform_object"):
            return self._to_1d_predictions(model_obj.predict(x_raw))
        x_trans = self._transform_for_raw_model(x_raw)
        return self._to_1d_predictions(model_obj.predict(x_trans))

    def evaluate_model(self) -> Evaluate_Model_Response:
        """
        Evaluate trained model against local baseline model (if available).
        """
        try:
            x, y_true = self._prepare_features_and_target()

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            y_hat_trained = self._predict_with_any_model(trained_model, x)
            trained_model_f1_score = f1_score(y_true, y_hat_trained, average="weighted")
            logging.info(f"Trained model F1 score on evaluation set: {trained_model_f1_score:.4f}")

            best_model_f1_score = None
            best_model, best_model_path = self.get_best_model(
                trained_model_path=self.model_trainer_artifact.trained_model_file_path
            )
            if best_model is not None:
                y_hat_best_model = self._predict_with_any_model(best_model, x)
                best_model_f1_score = f1_score(y_true, y_hat_best_model, average="weighted")
                logging.info(
                    f"Current baseline model F1 score on evaluation set: {best_model_f1_score:.4f} "
                    f"(path={best_model_path})"
                )
            else:
                logging.info("No local baseline model found. Accepting trained model by default.")

            tmp_best_model_score = 0.0 if best_model_f1_score is None else best_model_f1_score
            difference = trained_model_f1_score - tmp_best_model_score
            threshold = float(self.model_eval_config.changed_threshold_score)
            is_model_accepted = best_model_f1_score is None or difference > threshold

            result = Evaluate_Model_Response(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_model_accepted,
                difference=difference,
            )
            logging.info(f"Result: {result}")
            return result
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> Model_Evaluation_Artifact:
        """
        Initiate model evaluation and return artifact.
        """
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = Model_Evaluation_Artifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise CustomException(e, sys)
