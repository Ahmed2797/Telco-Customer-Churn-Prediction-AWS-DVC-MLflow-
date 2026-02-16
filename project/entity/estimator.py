import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from project.exception import CustomException



class ChurnTargetMapping:
    """
    Mapping for churn prediction target variable.
    """

    no_churn: int = 0
    churn: int = 1

    @classmethod
    def as_dict(cls) -> dict:
        """Return label → value mapping"""
        return {
            "no_churn": cls.no_churn,
            "churn": cls.churn,
        }

    @classmethod
    def reverse_mapping(cls) -> dict:
        """Return value → label mapping"""
        mapping = cls.as_dict()
        return {v: k for k, v in mapping.items()}



class ProjectModel:
    """
    Predict using a saved preprocessing pipeline and trained ML model.
    Accepts dict or DataFrame.
    """
    
    def __init__(self, transform_object, best_model_details: BaseEstimator | object):
        # Accept Pipeline, ColumnTransformer, or any sklearn-like transformer.
        if not hasattr(transform_object, "transform"):
            raise ValueError("transform_object must implement a transform() method")
        
        self.transform_object = transform_object
        self.best_model_details = best_model_details

    @staticmethod
    def _to_dataframe(data) -> pd.DataFrame:
        if isinstance(data, dict):
            dataframe = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            raise ValueError("Input must be a pandas DataFrame or a dict")

        if dataframe.empty:
            raise ValueError("Input DataFrame is empty")

        return dataframe

    def _get_model(self):
        return getattr(self.best_model_details, "best_model", self.best_model_details)

    def _prepare_dataframe(self, data) -> pd.DataFrame:
        """
        Prepare inference input:
        - strip whitespace in object columns
        - convert empty/null-like strings to NaN
        - convert numeric-like columns to numeric when all non-null values are numeric
        """
        dataframe = self._to_dataframe(data).copy()
        object_cols = dataframe.select_dtypes(include=["object"]).columns

        for col in object_cols:
            series = dataframe[col].astype(str).str.strip()
            series = series.replace(
                {"": np.nan, "na": np.nan, "nan": np.nan, "none": np.nan, "null": np.nan}
            )

            numeric_series = pd.to_numeric(series, errors="coerce")
            if numeric_series.notna().sum() == series.notna().sum():
                dataframe[col] = numeric_series
            else:
                dataframe[col] = series

        return dataframe

    @staticmethod
    def _collapse_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep inference feature engineering consistent with training-time postprocessing.
        """
        df_copy = df.copy()

        no_internet_cols = [col for col in df_copy.columns if "No internet service" in col]
        if no_internet_cols:
            df_copy["No_internet_service"] = (df_copy[no_internet_cols].sum(axis=1) > 0).astype(int)
            df_copy.drop(columns=no_internet_cols, inplace=True)

        if "MultipleLines_No phone service" in df_copy.columns:
            df_copy["No_phone_service"] = df_copy["MultipleLines_No phone service"].astype(int)
            df_copy.drop(columns=["MultipleLines_No phone service"], inplace=True)

        return df_copy

    def _postprocess_features(self, transformed_features):
        """
        Convert transformed matrix to named DataFrame and apply training-consistent
        feature collapsing before prediction.
        """
        if hasattr(self.transform_object, "get_feature_names_out"):
            feature_names = self.transform_object.get_feature_names_out()
            transformed_df = pd.DataFrame(transformed_features, columns=feature_names)
            transformed_df = self._collapse_redundant_columns(transformed_df)
            return transformed_df
        return transformed_features

    def _transform(self, data):
        dataframe = self._prepare_dataframe(data)
        transformed = self.transform_object.transform(dataframe)
        return self._postprocess_features(transformed)

    def predict(self, data):
        try:
            transformed_features = self._transform(data)
            model = self._get_model()
            # If model was fitted without feature names, pass numpy to avoid sklearn warning.
            model_input = transformed_features
            if isinstance(transformed_features, pd.DataFrame) and not hasattr(model, "feature_names_in_"):
                model_input = transformed_features.to_numpy()
            predictions = model.predict(model_input)
            return pd.DataFrame(predictions, columns=["prediction"])
        except Exception as e:
            raise CustomException(e, sys)

    def predict_proba(self, data):
        try:
            transformed_features = self._transform(data)
            model = self._get_model()

            if not hasattr(model, "predict_proba"):
                raise ValueError("Underlying model does not support predict_proba")

            model_input = transformed_features
            if isinstance(transformed_features, pd.DataFrame) and not hasattr(model, "feature_names_in_"):
                model_input = transformed_features.to_numpy()

            probabilities = model.predict_proba(model_input)
            if probabilities.shape[1] == 2:
                return pd.DataFrame(
                    probabilities,
                    columns=["proba_no_churn", "proba_churn"]
                )
            return pd.DataFrame(probabilities)
        except Exception as e:
            raise CustomException(e, sys)

    def predict_label(self, data):
        try:
            prediction_df = self.predict(data)
            reverse_map = ChurnTargetMapping.reverse_mapping()
            prediction_df["label"] = prediction_df["prediction"].map(reverse_map).fillna("unknown")
            return prediction_df
        except Exception as e:
            raise CustomException(e, sys)
