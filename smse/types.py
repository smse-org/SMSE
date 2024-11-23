from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from numpy.typing import NDArray

ImageT = NDArray[Union[np.float64, np.uint8]]
TextT = str


@dataclass
class AudioT:
    audio: List[torch.Tensor]
    sample_rate: int


@dataclass
class VideoT:
    frames: Union[ImageT, List[ImageT]]
    audio: AudioT
