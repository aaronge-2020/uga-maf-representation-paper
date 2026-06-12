"""Shared endpoint container used by manuscript benchmark runners."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EndpointData:
    name: str
    benchmark: str
    task: str
    labels: pd.Series | pd.DataFrame
    groups: pd.Series | None = None
