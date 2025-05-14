from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import torch
import torchaudio  # type: ignore[import-untyped,import-not-found]
import logging

from smse.pipelines.base import BaseConfig, BasePipeline
from smse.types import AudioT

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds


@dataclass
class AudioConfig(BaseConfig):
    sampling_rate: int = 16000
    max_duration: float = 30.0
    mono: bool = False
    normalize_audio: bool = False

    num_mel_bins: int = 128
    target_length: int = 204
    clip_duration: float = 2.0
    clips_per_audio: int = 3
    mean: float = -4.268
    std: float = 9.138
    use_clips: bool = False  # Whether to use clip sampling
    apply_melspec: bool = False  # Whether to convert to mel spectrogram


def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    """
    Convert a waveform to a mel-spectrogram.

    Args:
        waveform (torch.Tensor): Audio waveform
        sample_rate (int): Audio sample rate
        num_mel_bins (int): Number of mel bins
        target_length (int): Target length for the spectrogram

    Returns:
        torch.Tensor: Mel spectrogram
    """
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1)
    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames
    # if p is too large (say >20%), flash a warning
    if abs(p) / n_frames > 0.2:
        logging.warning(
            "Large gap between audio n_frames(%d) and "
            "target_length (%d). Is the audio_target_length "
            "setting correct?",
            n_frames,
            target_length,
        )
    # cut and pad
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
    # channel image
    fbank = fbank.unsqueeze(0)
    return fbank


def get_clip_timepoints(clip_sampler, duration):
    """
    Get the timepoints for clips from an audio file.

    Args:
        clip_sampler: A clip sampler object
        duration (float): Duration of the audio in seconds

    Returns:
        list: List of (start, end) timepoints
    """
    # Read out all clips in this video/audio
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


class AudioPipeline(BasePipeline):
    """
    Pipeline for processing audio data with options for compatibility with ImageBind.

    This pipeline provides two main processing modes:
    1. Simple processing: Basic audio processing with resampling, padding, etc.
    2. Clip-based processing: Similar to ImageBind, divides audio into clips and processes each.

    Both modes support optional transformation functions for custom processing steps.
    When configured with `use_clips=True` and `apply_melspec=True`, the pipeline
    directly produces tensors compatible with ImageBind's audio encoder.
    """

    def __init__(self, config: AudioConfig):
        """
        Initialize the audio pipeline with configuration.

        Args:
            config (AudioConfig): Configuration for audio processing
        """
        super().__init__(config)
        self.config: AudioConfig = config

    def load(self, input_paths: List[Path]) -> List[AudioT]:
        """
        Load audio data from a list of file paths.

        Args:
            input_paths (List[Path]): List of paths to the input audio files.

        Returns:
            List[AudioT]: List of loaded audio data.
        """
        audio_list = []
        for input_path in input_paths:
            waveform, sr = torchaudio.load(str(input_path))

            # Always allow resampling during loading
            if sr != self.config.sampling_rate:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr, new_freq=self.config.sampling_rate
                )

            audio_list.append(
                AudioT(audio=[waveform], sampling_rate=self.config.sampling_rate)
            )

        return audio_list

    def validate(self, data: Any) -> bool:
        return isinstance(data, AudioT)

    def _process_batch(
        self, audio_data_list: List[AudioT], transform_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Process a list of audio files with optional transform function and return a tensor.

        Args:
            audio_data_list (List[AudioT]): List of audio data to process
            transform_fn (Optional[Callable]): Optional transform function to apply to processed audio
                This function should take (waveform, sample_rate, config) and return a processed waveform

        Returns:
            torch.Tensor: Processed audio data as a tensor
                With shape [batch_size, clips_per_audio, 1, num_mel_bins, target_length]
        """
        # Process each audio item in the list
        processed_audio_tensors = []

        for audio_data in audio_data_list:
            if self.config.use_clips:
                processed = self._process_with_clips(audio_data, transform_fn)
            else:
                processed = self._process_simple(audio_data, transform_fn)

            # Stack the audio clips from each sample
            if processed.audio:
                audio_tensor = torch.stack(processed.audio, dim=0)
                processed_audio_tensors.append(audio_tensor)

        # Stack all processed audio into a single batch tensor
        if processed_audio_tensors:
            audio_batch = torch.cat(processed_audio_tensors, dim=0)
            # Move to specified device if provided
            return audio_batch.to(self.config.device)
        else:
            # Return empty tensor with correct shape if no audio was processed
            return torch.zeros(
                (
                    0,
                    self.config.clips_per_audio if self.config.use_clips else 1,
                    1,
                    self.config.num_mel_bins,
                    self.config.target_length,
                ),
                device=self.config.device,
            )

    def _process_simple(
        self, audio_data: AudioT, transform_fn: Optional[Callable]
    ) -> AudioT:
        """
        Process audio without clip sampling (original pipeline behavior).

        Args:
            audio_data: Single audio data item
            transform_fn: Optional transform function

        Returns:
            AudioT: Processed audio with consistent tensors ready for stacking
        """
        resampler = torchaudio.transforms.Resample(
            orig_freq=audio_data.sampling_rate,
            new_freq=self.config.sampling_rate,
        )

        target_length = int(self.config.max_duration * self.config.sampling_rate)

        processed_audio = []
        for waveform in audio_data.audio:
            # Add a channel dimension if the audio is mono (1D waveform)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            # Resample if needed
            if audio_data.sampling_rate != self.config.sampling_rate:
                waveform = resampler(waveform)

            # Convert to mono if specified
            if self.config.mono and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Normalize the waveform if specified
            if self.config.normalize_audio:
                waveform = waveform / waveform.abs().max()

            # Add padding or truncate to fixed length
            num_channels, waveform_length = waveform.shape
            if waveform_length > target_length:
                waveform = waveform[:, :target_length]
            elif waveform_length < target_length:
                padding = torch.zeros((num_channels, target_length - waveform_length))
                waveform = torch.cat([waveform, padding], dim=1)

            # Apply custom transform if provided
            if transform_fn is not None:
                waveform = transform_fn(
                    waveform, self.config.sampling_rate, self.config
                )

            # Apply mel spectrogram transformation if configured
            if self.config.apply_melspec:
                waveform = waveform2melspec(
                    waveform,
                    self.config.sampling_rate,
                    self.config.num_mel_bins,
                    self.config.target_length,
                )

                # For consistency with the clips version, normalize if configured
                if not self.config.normalize_audio:
                    from torchvision.transforms import Normalize

                    normalize = Normalize(mean=self.config.mean, std=self.config.std)
                    waveform = normalize(waveform)

                # Add a dimension to match the clip-based output shape
                # [1, num_mel_bins, target_length] → [1, 1, num_mel_bins, target_length]
                waveform = waveform.unsqueeze(0)

            processed_audio.append(waveform)

        return AudioT(audio=processed_audio, sampling_rate=self.config.sampling_rate)

    def __call__(
        self,
        input_data: Union[List[Path], List[AudioT]],
        transform_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Process audio data directly from paths or AudioT objects.

        Args:
            input_data (Union[List[Path], List[AudioT]]): Input audio data or paths
            transform_fn (Optional[Callable]): Optional transform function

        Returns:
            torch.Tensor: Processed audio tensor
        """
        data: List[AudioT]
        if all(isinstance(item, Path) for item in input_data):
            data = self.load(input_data)
        else:
            data = input_data

        return self.process(data, transform_fn)

    # Override the parent method to ensure compatibility with the base class
    def process(
        self, data: Any, transform_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Process audio data and return a tensor.

        Args:
            data: Audio data to process (either a single AudioT or List[AudioT])
            transform_fn: Optional transform function

        Returns:
            torch.Tensor: Processed audio tensor
        """
        # Handle both single AudioT and List[AudioT] cases
        if isinstance(data, AudioT):
            return self._process_batch([data], transform_fn)
        elif isinstance(data, list) and all(isinstance(item, AudioT) for item in data):
            return self._process_batch(data, transform_fn)
        else:
            raise ValueError(f"Expected AudioT or List[AudioT], got {type(data)}")

    def _process_with_clips(
        self, audio_data: AudioT, transform_fn: Optional[Callable]
    ) -> AudioT:
        """
        Process audio with clip sampling.
        """
        from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler

        audio_outputs = []

        for waveform in audio_data.audio:
            # Add a channel dimension if the audio is mono (1D waveform)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            # Convert to mono if specified
            if self.config.mono or waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Create clip sampler
            clip_sampler = ConstantClipsPerVideoSampler(
                clip_duration=self.config.clip_duration,
                clips_per_video=self.config.clips_per_audio,
            )

            # Get clip timepoints
            duration = waveform.size(1) / self.config.sampling_rate
            all_clips_timepoints = get_clip_timepoints(clip_sampler, duration)

            all_clips = []
            # Process each clip
            for clip_timepoints in all_clips_timepoints:
                waveform_clip = waveform[
                    :,
                    int(clip_timepoints[0] * self.config.sampling_rate) : int(
                        clip_timepoints[1] * self.config.sampling_rate
                    ),
                ]

                # Apply custom transform if provided
                if transform_fn is not None:
                    waveform_clip = transform_fn(
                        waveform_clip, self.config.sampling_rate, self.config
                    )

                # Apply mel spectrogram if configured
                if self.config.apply_melspec:
                    waveform_clip = waveform2melspec(
                        waveform_clip,
                        self.config.sampling_rate,
                        self.config.num_mel_bins,
                        self.config.target_length,
                    )

                    # For compatibility with ImageBind, normalize the mel spectrogram
                    from torchvision.transforms import Normalize

                    normalize = Normalize(mean=self.config.mean, std=self.config.std)
                    waveform_clip = normalize(waveform_clip)

                all_clips.append(waveform_clip)

            # Stack all clips for this audio sample
            if all_clips:
                stacked_clips = torch.stack(all_clips, dim=0)
                audio_outputs.append(stacked_clips)

        # Combine all audio outputs
        if audio_outputs:
            return AudioT(audio=audio_outputs, sampling_rate=self.config.sampling_rate)
        else:
            return AudioT(audio=[], sampling_rate=self.config.sampling_rate)
