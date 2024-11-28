from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import torch
import torchaudio  # type: ignore[import-untyped,import-not-found]

from smse.pipelines.base import BaseConfig, BasePipeline
from smse.types import AudioT


@dataclass
class AudioConfig(BaseConfig):
    sample_rate: int = 16000
    max_duration: float = 30.0
    mono: bool = False
    normalize_audio: bool = False


class AudioPipeline(BasePipeline):
    def __init__(self, config: AudioConfig):
        super().__init__(config)
        self.config: AudioConfig = config

    def load(self, input_path: Union[str, Path]) -> AudioT:
        """
        Load audio data. If `input_path` is a directory, load all audio
        files in the directory.
        """
        input_path = Path(input_path)
        sr = 0

        if input_path.is_dir():
            # Load all audio files in the directory
            audio_files = list(
                input_path.glob("*.wav")
            )  # Adjust glob pattern for required extensions
            audio_list = []
            for file in audio_files:
                waveform, sr = torchaudio.load(str(file))
                audio_list.append(waveform)
        else:
            # Load a single audio file
            waveform, sr = torchaudio.load(str(input_path))
            audio_list = [waveform]

        # Validate sample rates for consistency
        sample_rate = sr
        for waveform in audio_list:
            if sr != waveform.shape[1]:
                raise ValueError("Inconsistent sample rates in audio files")

        return AudioT(audio=audio_list, sample_rate=sample_rate)

    def validate(self, data: Any) -> bool:
        return isinstance(data, AudioT)

    def process(self, audio_data: AudioT) -> AudioT:
        """
        Process a batch of audio files.
        """
        processed_audio = []
        for waveform in audio_data.audio:
            # Resample if needed
            if audio_data.sample_rate != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=audio_data.sample_rate,
                    new_freq=self.config.sample_rate,
                )
                waveform = resampler(waveform)

            # Convert to mono if specified
            if self.config.mono and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Normalize the waveform if specified
            if self.config.normalize_audio:
                waveform = waveform / torch.max(torch.abs(waveform))

            # Add padding or truncate to fixed length
            target_length = int(self.config.max_duration * self.config.sample_rate)
            if waveform.shape[0] > target_length:
                waveform = waveform[:target_length]
            elif waveform.shape[0] < target_length:
                padding = torch.zeros((target_length - waveform.shape[0]))
                waveform = torch.cat([waveform, padding], dim=0)

            processed_audio.append(waveform)

        return AudioT(audio=processed_audio, sample_rate=self.config.sample_rate)
