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
