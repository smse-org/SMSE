from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Union

import torch
import torchaudio  # type: ignore[import-untyped,import-not-found]

from smse.pipelines.base import BaseConfig, BasePipeline
from smse.types import AudioT


@dataclass
class AudioConfig(BaseConfig):
    sampling_rate: int = 16000
    max_duration: float = 30.0
    mono: bool = False
    normalize_audio: bool = False


class AudioPipeline(BasePipeline):
    def __init__(self, config: AudioConfig):
        super().__init__(config)
        self.config: AudioConfig = config

    def load(self, input_paths: Sequence[Union[str, Path]]) -> List[AudioT]:
        """
        Load audio data from a list of file paths.

        Args:
            input_paths (List[Union[str, Path]]): List of paths to the input audio files.

        Returns:
            AudioT: Loaded audio data.
        """
        audio_list = []
        for input_path in input_paths:
            waveform, sr = torchaudio.load(str(input_path))

            if sr != self.config.sampling_rate:
                raise ValueError(
                    f"Sample rate of {input_path} is {sr}, "
                    "but expected {self.config.sampling_rate}"
                )

            audio_list.append(
                AudioT(audio=waveform, sampling_rate=self.config.sampling_rate)
            )

        return audio_list

    def process(self, audio_data: AudioT) -> AudioT:
        """
        Process a batch of audio files.
        """
        resampler = torchaudio.transforms.Resample(
            orig_freq=audio_data.sampling_rate,
            new_freq=self.config.sampling_rate,
        )

        target_length = int(self.config.max_duration * self.config.sampling_rate)
        channels = audio_data.data

        # Add a channel dimension if the audio is mono (1D waveform)
        if channels.dim() == 1:
            channels = channels.unsqueeze(0)

        # Convert to mono if specified
        if self.config.mono and channels.shape[0] > 1:
            channels = channels.mean(dim=0, keepdim=True)

        processed_audio = []
        for channel in channels:
            # Resample if needed
            if audio_data.sampling_rate != self.config.sampling_rate:
                channel = resampler(channel)

            # Normalize the waveform if specified
            if self.config.normalize_audio:
                channel = channel / channel.abs().max()

            # Add padding or truncate to fixed length
            waveform_length = channel.shape
            if waveform_length > target_length:
                channel = channel[:target_length]
            elif waveform_length < target_length:
                padding = torch.zeros((target_length - waveform_length))
                channel = torch.cat([channel, padding], dim=1)

            processed_audio.append(channel)

        return AudioT(audio=processed_audio, sampling_rate=self.config.sampling_rate)
