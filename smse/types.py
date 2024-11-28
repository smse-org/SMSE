from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray
from PIL import Image

AudioArrayT = NDArray[np.float32]
ImageT = NDArray[np.float64 | np.uint8] | Image.Image
TextT = str


@dataclass
class AudioT:
    audio: AudioArrayT
    sample_rate: int


@dataclass
class VideoT:
    frames: List[ImageT]
    audio: AudioT
