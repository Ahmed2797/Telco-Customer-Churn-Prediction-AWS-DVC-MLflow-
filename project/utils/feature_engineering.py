import pandas as pd
import numpy as np

def encode_binary(x):
    """
    Universal gender encoder.
    Works for:
    - pandas Series
    - single value
    """

    # Case 1 → Pandas Series (training data)
    if isinstance(x, pd.Series):
        return (
            x.astype(str)
             .str.strip()
             .str.lower()
             .eq("male")
             .astype(int)
        )

    # Case 2 → Single value (prediction)
    return int(str(x).strip().lower() == "Male")


# -------------------------------------------------
# Collapse Redundant Columns (AFTER ENCODING)
# -------------------------------------------------
def collapse_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    df_copy = df.copy()

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


def empty_string_columns(df: pd.DataFrame) -> pd.DataFrame:
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