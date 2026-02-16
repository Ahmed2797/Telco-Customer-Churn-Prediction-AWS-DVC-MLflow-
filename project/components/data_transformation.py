import os
import sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.combine import SMOTEENN

from project.entity.config import Data_Transformation_Config
from project.entity.artifacts import (
    Data_Ingestion_Artifact,
    Data_Validation_Artifact,
    Data_Transformation_Artifact
)
from project.constants import (
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
    COLUMN_YAML_FILE_PATH,
    Target_Column
)
from project.utils import read_yaml, save_object, save_numpy_array, remove_high_vif_features
from project.exception import CustomException
from project.logger import logging


class Data_Transformation:
    """
    Handles all preprocessing steps for customer churn data, including:

    - Binary mapping for Yes/No and Male/Female features
    - Log transformation of specified numerical columns
    - Collapse redundant dummy variables
    - Preprocessing using ColumnTransformer
    - Handling imbalanced data with SMOTEENN
    - Saving processed train/test arrays and preprocessing object

    Parameters
    ----------
    data_transformation_config : Data_Transformation_Config
        Configuration object containing paths for saving artifacts
    data_ingestion_artifact : Data_Ingestion_Artifact
        Artifact containing paths of ingested train/test data
    data_validation_artifact : Data_Validation_Artifact
        Artifact from data validation step
    """

    def __init__(self,
                 data_transformation_config: Data_Transformation_Config,
                 data_ingestion_artifact: Data_Ingestion_Artifact,
                 data_validation_artifact: Data_Validation_Artifact):
        """
        Initializes the Data_Transformation object and reads column schema from YAML.
        """
        try:
            self.transformation_config = data_transformation_config
            self.ingestion_artifact = data_ingestion_artifact
            self.validation_artifact = data_validation_artifact
            self._column_schema = read_yaml(COLUMN_YAML_FILE_PATH)

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------
    # Binary Mapping
    # -------------------------------------------------
    def apply_binary_mapping(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies binary mapping to categorical columns with 2 unique values
        (e.g., Yes/No, Male/Female) as defined in the YAML schema.

        Parameters
        ----------
        train_df : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            DataFrame with binary columns mapped to 0/1
        """
        df_copy = train_df.copy()
        binary_cols = self._column_schema.get("binary_categorical_columns", {})

        for col, details in binary_cols.items():
            if col in df_copy.columns:
                mapping = details.get("mapping", {})
                df_copy[col] = df_copy[col].replace(mapping)

        return df_copy

    def empty_string_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strip string values and convert blank/whitespace-only strings to NaN.
        """
        df_copy = df.copy()
        object_cols = df_copy.select_dtypes(include=["object"]).columns
        for col in object_cols:
            df_copy[col] = df_copy[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
            df_copy[col] = df_copy[col].replace("", np.nan)
        return df_copy

    # -------------------------------------------------
    # Log Transform
    # -------------------------------------------------
    def apply_log_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies log1p transformation to columns specified in the YAML schema.

        Parameters
        ----------
        train_df : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            DataFrame with specified columns log-transformed
        """
        df_copy = train_df.copy()
        log_cols = self._column_schema.get("log_transform_col", [])

        for col in log_cols:
            if col in df_copy.columns:
                df_copy[col] = np.log1p(df_copy[col].clip(lower=0))

        return df_copy

    # -------------------------------------------------
    # Collapse Redundant Columns (AFTER ENCODING)
    # -------------------------------------------------
    def collapse_redundant_columns(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapses redundant dummy variables into single columns to reduce multicollinearity.
        Example:
        - Combines all 'No internet service' dummy columns into one 'No_internet_service'
        - Converts 'MultipleLines_No phone service' into 'No_phone_service'

        Parameters
        ----------
        train_df : pd.DataFrame
            Input dataframe after one-hot encoding

        Returns
        -------
        pd.DataFrame
            DataFrame with collapsed redundant columns
        """
        df_copy = train_df.copy()

        # Collapse "No internet service"
        no_internet_cols = [
            col for col in df_copy.columns
            if "No internet service" in col
        ]

        if no_internet_cols:
            df_copy["No_internet_service"] = (
                df_copy[no_internet_cols].sum(axis=1) > 0
            ).astype(int)

            df_copy.drop(columns=no_internet_cols, inplace=True)

        # Collapse "No phone service"
        if "MultipleLines_No phone service" in df_copy.columns:
            df_copy["No_phone_service"] = df_copy[
                "MultipleLines_No phone service"
            ].astype(int)

            df_copy.drop(columns=["MultipleLines_No phone service"], inplace=True)

        return df_copy

    # -------------------------------------------------
    # Preprocessor
    # -------------------------------------------------
    def get_preprocessor(self):
        """
        Creates a ColumnTransformer preprocessing pipeline for numeric and categorical features.

        Returns
        -------
        ColumnTransformer
            Preprocessing pipeline
        """
        numerical_cols = self._column_schema.get("numerical_columns", [])
        multi_cat_cols = self._column_schema.get("multi_categorical_columns", [])

        numeric_pipeline = Pipeline([
            ("imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, multi_cat_cols)
        ])

        return preprocessor

    # -------------------------------------------------
    # Main Transformation
    # -------------------------------------------------
    def initiate_data_transformation(self) -> Data_Transformation_Artifact:
        """
        Performs the full data transformation pipeline:
        - Drop unnecessary columns
        - Apply binary mapping
        - Convert TotalCharges to numeric
        - Apply log transform
        - Preprocess with ColumnTransformer
        - Collapse redundant columns
        - Optionally calculate VIF (currently commented)
        - Apply SMOTEENN to balance classes
        - Save train/test arrays and preprocessing object

        Returns
        -------
        Data_Transformation_Artifact
            Artifact containing paths to processed train/test arrays and preprocessing object
        """
        try:
            logging.info("Reading train and test data")

            train_df = pd.read_csv(self.ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.ingestion_artifact.test_file_path)

            # Normalize raw string values early (e.g., ' ' -> NaN)
            train_df = self.empty_string_columns(train_df)
            test_df = self.empty_string_columns(test_df)

            # Drop columns
            drop_cols = self._column_schema.get("drop_columns", [])
            train_df.drop(columns=drop_cols, inplace=True, errors="ignore")
            test_df.drop(columns=drop_cols, inplace=True, errors="ignore")

            # Binary mapping
            train_df = self.apply_binary_mapping(train_df)
            test_df = self.apply_binary_mapping(test_df)

            # Convert TotalCharges
            if "TotalCharges" in train_df.columns and "TotalCharges" in test_df.columns:
                train_df["TotalCharges"] = pd.to_numeric(train_df["TotalCharges"], errors="coerce")
                test_df["TotalCharges"] = pd.to_numeric(test_df["TotalCharges"], errors="coerce")

                train_df["TotalCharges"] = train_df["TotalCharges"].fillna(train_df["MonthlyCharges"])
                test_df["TotalCharges"] = test_df["TotalCharges"].fillna(test_df["MonthlyCharges"])


            # Log transform
            train_df = self.apply_log_transform(train_df)
            test_df = self.apply_log_transform(test_df)

            # Separate target
            X_train = train_df.drop(columns=[Target_Column])
            y_train = train_df[Target_Column]


            X_test = test_df.drop(columns=[Target_Column])
            y_test = test_df[Target_Column]

            # Check train == test columns
            assert set(X_train.columns) == set(X_test.columns)
            logging.info(f"Missing values:\n{X_train.isnull().sum()}")

            # Preprocess
            preprocessor = self.get_preprocessor()
            X_train_trans = preprocessor.fit_transform(X_train)
            X_test_trans = preprocessor.transform(X_test)

            # Get feature names properly
            feature_names = preprocessor.get_feature_names_out()
            X_train_trans = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
            X_test_trans = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)
            logging.info(f"Total transformed features: {len(X_train_trans.columns)}")

            # Collapse redundant columns
            X_train_trans = self.collapse_redundant_columns(X_train_trans)
            X_test_trans = self.collapse_redundant_columns(X_test_trans)

            # # Optional: Remove high VIF features (commented for now)
            # X_train_trans = self.remove_high_vif_features(X_train_trans, threshold=5.0)
            # X_test_trans = X_test_trans[X_train_trans.columns]

            # # Apply SMOTEENN
            # logging.info("Applying SMOTEENN")
            # smote = SMOTEENN(sampling_strategy="minority", random_state=42)
            # X_train_resampled, y_train_resampled = smote.fit_resample(
            #     X_train_trans, y_train
            # )
            # # Combine train/test arrays
            # train_arr = np.c_[X_train_resampled, np.array(y_train_resampled)]
            # test_arr = np.c_[X_test_trans, np.array(y_test)]

            train_arr = np.c_[X_train_trans, np.array(y_train)]
            test_arr = np.c_[X_test_trans, np.array(y_test)]

            # Save arrays and preprocessor object
            save_numpy_array(self.transformation_config.transform_train_path, train_arr)
            save_numpy_array(self.transformation_config.transform_test_path, test_arr)
            save_object(self.transformation_config.transform_object_path, preprocessor)

            logging.info("Data Transformation Completed Successfully")

            return Data_Transformation_Artifact(
                transform_train_path=self.transformation_config.transform_train_path,
                transform_test_path=self.transformation_config.transform_test_path,
                preprocessing_pkl=self.transformation_config.transform_object_path
            )

        except Exception as e:
            raise CustomException(f"Error in initiate_data_transformation: {e}", sys)
