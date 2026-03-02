from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence


@dataclass(frozen=True)
class Config:
    """
    Central configuration for gene benchmark pipelines.

    Defaults are enforced globally:
        dga = "disgenet"
        time = 2019
        time_split = True
    """

    # Required defaults
    dga: str = "disgenet"
    time: int = 2019
    time_split: bool = True

    # Paths
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(init=False)
    results_dir: Path = field(init=False)

    # Feature selection
    feature_list: Sequence[str] = field(
        default_factory=lambda: [
            "uniport_ppi_2019",
            "ppi_2019_dw_40",
            "uniport_bio",
            "uniport_seq",
            "uniport_esm",
            "diffusion_2019",
        ]
    )

    # Parallelism
    num_processes: int = 20

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", self.root / "data")
        object.__setattr__(self, "results_dir", self.root / "results")


def default_config() -> Config:
    """Convenience factory to obtain the enforced global defaults."""
    return Config()
