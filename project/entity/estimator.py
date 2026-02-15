import sys
import pandas as pd
from sklearn.pipeline import Pipeline
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
    
    def __init__(self, transform_object: Pipeline, best_model_details: BaseEstimator | object):
        if not isinstance(transform_object, Pipeline):
            raise ValueError("transform_object must be a scikit-learn Pipeline")
        
        self.transform_object = transform_object
        self.best_model_details = best_model_details

    def predict(self, data):
        dataframe = None  # Prevent UnboundLocalError

        try:
            # Convert dict → DataFrame
            if isinstance(data, dict):
                dataframe = pd.DataFrame([data])

            # Already a DataFrame
            elif isinstance(data, pd.DataFrame):
                dataframe = data

            else:
                raise ValueError("Input must be a pandas DataFrame or a dict")

            # Check empty
            if dataframe.empty:
                raise ValueError("Input DataFrame is empty")

            # Transform features
            transformed_features = self.transform_object.transform(dataframe)

            # Extract model (in case it's wrapped)
            model = getattr(self.best_model_details, "best_model", self.best_model_details)

            # Predict
            predictions = model.predict(transformed_features)

            # Return DataFrame
            return pd.DataFrame(predictions, columns=["prediction"])

        except Exception as e:
            raise CustomException(e, sys)
