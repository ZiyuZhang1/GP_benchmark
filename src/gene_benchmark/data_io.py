from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from .config import Config


def load_dga(config: Config) -> pd.DataFrame:
    """
    Load disease–gene association table based on config.dga.
    """
    if config.dga == "disgenet":
        path = config.data_dir / "disgent_2020/timecut/dga_time_uniport.csv"
    elif config.dga == "opentarget":
        path = config.data_dir / "opentarget/ot_dga_time_uni.csv"
    else:
        raise ValueError(f"Unsupported dga source: {config.dga}")

    df = pd.read_csv(path)
    if config.dga == "opentarget":
        df = df[df["score"] >= 0.4]
    return df


def load_edge_list(config: Config, time: int | None = None) -> Path:
    """
    Return path to edge CSV for graph models.
    """
    year = time or config.time
    path = config.data_dir / f"stringdb/edge_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Edge list not found: {path}")
    return path
