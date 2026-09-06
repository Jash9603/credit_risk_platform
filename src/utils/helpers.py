"""Small utilities shared across data/ml/talk_to_data modules."""
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def timer(label: str):
    start = time.perf_counter()
    yield
    logger.info("%s took %.2fs", label, time.perf_counter() - start)


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to the smallest safe dtype. The Home Credit tables load
    as float64/int64 by default and comfortably exceed a few GB combined; downcasting
    is what keeps the full multi-table join workable on a laptop."""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype
        # pandas 3's default text dtype isn't plain `object` anymore, so check numeric-ness
        # directly rather than assuming "not object" means numeric.
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        c_min, c_max = df[col].min(), df[col].max()
        if str(col_type)[:3] == "int":
            for dtype in (np.int8, np.int16, np.int32, np.int64):
                info = np.iinfo(dtype)
                if info.min <= c_min and c_max <= info.max:
                    df[col] = df[col].astype(dtype)
                    break
        else:
            for dtype in (np.float32, np.float64):
                info = np.finfo(dtype)
                if info.min <= c_min and c_max <= info.max:
                    df[col] = df[col].astype(dtype)
                    break

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        logger.info(
            "Memory usage reduced: %.2f MB -> %.2f MB (-%.1f%%)",
            start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem,
        )
    return df
