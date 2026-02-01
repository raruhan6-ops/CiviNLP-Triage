import pandas as pd
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

def load_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    logger.info(f"Loading CSV from {path}")
    return pd.read_csv(path, **kwargs)

def save_parquet(df: pd.DataFrame, path: Union[str, Path], index: bool = False, **kwargs) -> None:
    """Save a pandas DataFrame to a Parquet file, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving Parquet to {path}")
    df.to_parquet(path, index=index, **kwargs)

def load_parquet(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load a Parquet file into a pandas DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    logger.info(f"Loading Parquet from {path}")
    return pd.read_parquet(path, **kwargs)
