from dataclasses import dataclass
from typing import List, Union

import numpy as np
from numpy.typing import NDArray

AudioArrayT = NDArray[np.float32]
ImageT = NDArray[Union[np.float64, np.uint8]]
TextT = str


@dataclass
class AudioT:
    audio: AudioArrayT
    sample_rate: int


@dataclass
class VideoT:
    frames: Union[ImageT, List[ImageT]]
    audio: AudioT
