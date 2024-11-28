from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

ImageT = NDArray[np.float64 | np.uint8] | Image.Image | torch.Tensor
TextT = str


@dataclass
class AudioT:
    audio: List[torch.Tensor]
    sampling_rate: int


@dataclass
class VideoT:
    frames: List[ImageT] | torch.Tensor
    audio: AudioT
