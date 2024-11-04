from dataclasses import dataclass
from typing import Union, Any
from pathlib import Path
from pipelines.base_pipeline import PipelineConfig, Pipeline
import numpy as np


@dataclass
class AudioConfig(PipelineConfig):
    sample_rate: int = 16000
    max_duration: float = 30.0
    mono: bool = True
    normalize_audio: bool = True


# Audio Pipeline
class AudioPipeline(Pipeline):
    def __init__(self, config: AudioConfig):
        super().__init__(config)
        self.config: AudioConfig = config

    def load(self, input_path: Union[str, Path]) -> tuple:
        """Load audio from file"""
        try:
            import librosa

            audio, sr = librosa.load(
                input_path,
                sr=self.config.sample_rate,
                mono=self.config.mono,
                duration=self.config.max_duration,
            )
            return audio, sr
        except ImportError:
            raise ImportError("librosa is required for audio processing")

    def validate(self, data: Any) -> bool:
        return isinstance(data, tuple) and len(data) == 2

    def preprocess(self, audio_data: tuple) -> np.ndarray:
        """Preprocess audio data"""
        audio, sr = audio_data

        if self.config.normalize_audio:
            try:
                import librosa

                audio = librosa.util.normalize(audio)
            except ImportError:
                raise ImportError("librosa is required for audio processing")

        # Add padding or truncate to fixed length if needed
        target_length = int(self.config.max_duration * sr)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))

        return audio
