from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np

from smse.pipelines.base import BasePipeline, PipelineConfig
from smse.types import AudioT


@dataclass
class AudioConfig(PipelineConfig):
    sample_rate: int = 16000
    max_duration: float = 30.0
    mono: bool = True
    normalize_audio: bool = True


class AudioPipeline(BasePipeline):
    def __init__(self, config: AudioConfig):
        super().__init__(config)
        self.config: AudioConfig = config

    def load(self, input_path: Union[str, Path]) -> AudioT:
        """Load audio from file"""
        try:
            import librosa  # type: ignore[import-not-found]

            audio, sr = librosa.load(
                input_path,
                sr=self.config.sample_rate,
                mono=self.config.mono,
                duration=self.config.max_duration,
            )
            return AudioT(audio=audio, sample_rate=sr)
        except ImportError:
            raise ImportError("librosa is required for audio processing")

    def validate(self, data: Any) -> bool:
        return isinstance(data, AudioT)

    def process(self, audio_data: AudioT) -> AudioT:
        """Preprocess audio data"""
        audio, sr = audio_data.audio, audio_data.sample_rate

        if self.config.normalize_audio:
            try:
                import librosa  # type: ignore[import-not-found]

                audio = librosa.util.normalize(audio)
            except ImportError:
                raise ImportError("librosa is required for audio processing")

        # Add padding or truncate to fixed length if needed
        target_length = int(self.config.max_duration * sr)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))

        return AudioT(audio=audio, sample_rate=sr)
