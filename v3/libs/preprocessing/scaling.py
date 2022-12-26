from sklearn.preprocessing import minmax_scale
import pandas as pd
import numpy as np

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize data to range [0, 1] with given scaler (default: MinMaxScaler)
    """
    return minmax_scale(data)


def normalize_columns(df: pd.DataFrame, columns: list[str], inplace=False) -> pd.DataFrame:
    """
    Normalize given pandas columns to range [0, 1] with given scaler (default: MinMaxScaler)
    """
    df = df if inplace else df.copy()
    for column in columns:
        df[column] = normalize_data(df[column].to_numpy().reshape(-1, 1))
    return df