import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets")


def get_dataset_names() -> list[str]:
    """Return a list of available dataset names (CSV files without extension)."""
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' does not exist.")
    dataset_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    return sorted({f[:-4] for f in dataset_files})


def get_dataframe(name: str) -> pd.DataFrame:
    """Load a dataset as a raw DataFrame."""
    dataset_path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset '{name}' not found in '{DATA_DIR}'.")
    return pd.read_csv(dataset_path)


def build_survival_y(
    df: pd.DataFrame,
    event_col: str = "event",
    time_col: str = "time",
) -> np.ndarray:
    """
    Build a structured array y with dtype [('event', bool), ('time', float)] from a raw DataFrame.
    """
    if event_col not in df.columns or time_col not in df.columns:
        raise ValueError(f"Missing required columns: {event_col!r}, {time_col!r}")

    y = np.zeros(len(df), dtype=[("event", bool), ("time", float)])
    y["event"] = df[event_col].astype(bool).to_numpy()
    y["time"] = df[time_col].astype(float).to_numpy()
    return y
