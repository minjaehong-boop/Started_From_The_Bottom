from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass(frozen=True)
class Box:
    x: int; y: int; w: int; h: int

@dataclass(frozen=True)
class TileTask:
    frame_id: int
    tile_id: int
    box: Box
    image: np.ndarray
    t_enqueue: float

@dataclass(frozen=True)
class TileResult:
    frame_id: int
    tile_id: int
    box: Box
    image: np.ndarray
