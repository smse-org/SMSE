from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Union

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

    def load(self, input_path: Union[str, Path]) -> VideoT:
        """Load video from file"""
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
            audio_data: AudioT = self.audio_pipeline.load(input_path)

            return VideoT(frames=frames, audio=audio_data)
        except ImportError:
            raise ImportError("OpenCV is required for video processing")

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

        return VideoT(frames=[processed_frames], audio=processed_audio)
