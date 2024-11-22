from dataclasses import dataclass
from typing import List, Union

import numpy as np
from PIL import Image
from numpy.typing import NDArray

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
