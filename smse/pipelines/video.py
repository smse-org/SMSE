from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import numpy as np

from smse.pipelines.audio import AudioConfig, AudioPipeline
from smse.pipelines.base import BaseConfig, BasePipeline
from smse.pipelines.image import ImageConfig, ImagePipeline
from smse.types import AudioT, ImageT, VideoT


@dataclass
class VideoConfig(BaseConfig):
    fps: int = 30
    max_frames: int = 32
    image_config: ImageConfig = field(default_factory=ImageConfig)
    audio_config: AudioConfig = field(default_factory=AudioConfig)


# Video Pipeline
class VideoPipeline(BasePipeline):
    def __init__(self, config: VideoConfig):
        super().__init__(config)
        self.config: VideoConfig = config
        self.image_pipeline = ImagePipeline(config.image_config)
        self.audio_pipeline = AudioPipeline(config.audio_config)

    def load(self, input_paths: List[Path]) -> List[VideoT]:
        """
        Load video data from a list of file paths.

        Args:
            input_paths (List[Path]): List of paths to the input video files.

        Returns:
            List[VideoT]: Loaded video data.
        """
        videos = []
        for input_path in input_paths:
            try:
                import cv2  # type: ignore[import-not-found]

                cap = cv2.VideoCapture(str(input_path))
                frames: List[ImageT] = []
                while len(frames) < self.config.max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()

                # Load audio using AudioPipeline
                audio_data: AudioT = self.audio_pipeline.load([input_path])[0]

                videos.append(
                    VideoT(frames=frames, audio=audio_data, fps=self.config.fps)
                )
            except ImportError:
                raise ImportError("OpenCV is required for video processing")
        return videos

    def validate(self, data: Any) -> bool:
        return isinstance(data, VideoT)

    def process(self, video_data: VideoT) -> VideoT:
        """Preprocess video data"""
        frames = video_data.frames

        # Sample frames if needed
        if len(frames) > self.config.max_frames:
            indices = np.linspace(0, len(frames) - 1, self.config.max_frames, dtype=int)
            frames = [frames[i] for i in indices]

        # Process frames using ImagePipeline
        processed_frames = self.image_pipeline.process(frames)

        # Process audio if available
        processed_audio = None
        if video_data.audio is not None:
            processed_audio = self.audio_pipeline.process(video_data.audio)

        return VideoT(
            frames=[processed_frames],
            audio=AudioT(
                [processed_audio], sampling_rate=self.config.audio_config.sampling_rate
            ),
            fps=self.config.fps,
        )
