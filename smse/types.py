from dataclasses import dataclass
from enum import Enum, auto
from typing import List

import numpy as np
import torch
from numpy import ndarray
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor


class Modality(Enum):
    """Supported modalities in the SMSE framework."""

    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()


ImageT = NDArray[np.float64 | np.uint8] | Image.Image | torch.Tensor
TextT = str
EmbeddingT = List[Tensor] | ndarray | Tensor


@dataclass
class AudioT:
    audio: List[torch.Tensor]
    sampling_rate: int


@dataclass
class VideoT:
    frames: List[ImageT]
    audio: AudioT
