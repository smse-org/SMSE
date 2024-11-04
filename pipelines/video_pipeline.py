from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any, Callable
from pathlib import Path
import numpy as np
import logging
from enum import Enum
from pipelines.base_pipeline import PipelineConfig, DataType, Pipeline
from pipelines.image_pipeline import ImageConfig, ImagePipeline
from pipelines.audio_pipeline import AudioConfig, AudioPipeline


@dataclass
class VideoConfig(PipelineConfig):
    fps: int = 30
    max_frames: int = 32
    image_config: ImageConfig = ImageConfig(input_type=DataType.IMAGE)
    audio_config: AudioConfig = AudioConfig(input_type=DataType.AUDIO)


# Video Pipeline
class VideoPipeline(Pipeline):
    def __init__(self, config: VideoConfig):
        super().__init__(config)
        self.config: VideoConfig = config
        self.image_pipeline = ImagePipeline(config.image_config)
        self.audio_pipeline = AudioPipeline(config.audio_config)

    def load(self, input_path: Union[str, Path]) -> Dict[str, Any]:
        """Load video from file"""
        try:
            import cv2

            cap = cv2.VideoCapture(str(input_path))
            frames = []
            while len(frames) < self.config.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            # Load audio using AudioPipeline
            audio_data = self.audio_pipeline.load(input_path)

            return {"frames": frames, "audio": audio_data}
        except ImportError:
            raise ImportError("OpenCV is required for video processing")

    def validate(self, data: Any) -> bool:
        return isinstance(data, dict) and "frames" in data

    def preprocess(self, video_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Preprocess video data"""
        frames = video_data["frames"]

        # Sample frames if needed
        if len(frames) > self.config.max_frames:
            indices = np.linspace(0, len(frames) - 1, self.config.max_frames, dtype=int)
            frames = [frames[i] for i in indices]

        # Process frames using ImagePipeline
        processed_frames = np.stack(
            [self.image_pipeline.preprocess(frame) for frame in frames]
        )

        # Process audio if available
        processed_audio = None
        if "audio" in video_data and video_data["audio"] is not None:
            processed_audio = self.audio_pipeline.preprocess(video_data["audio"])

        return {"frames": processed_frames, "audio": processed_audio}
