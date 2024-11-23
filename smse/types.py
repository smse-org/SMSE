from dataclasses import dataclass
from typing import List, Union

import torch
import numpy as np
from numpy.typing import NDArray

ImageT = NDArray[Union[np.float64, np.uint8]]
TextT = str


@dataclass
class AudioT:
    audio: torch.Tensor
    sample_rate: int


@dataclass
class VideoT:
    frames: Union[ImageT, List[ImageT]]
    audio: AudioT
