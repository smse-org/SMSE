from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from smse.pipelines.audio import AudioConfig, AudioPipeline
from smse.pipelines.image import ImageConfig, ImagePipeline
from smse.pipelines.multimodal import MultimodalPipeline, PipelineMapping
from smse.pipelines.text import TextConfig, TextPipeline
from smse.types import AudioT, ImageT, TextT


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        pipeline: MultimodalPipeline,
        data_loader: DataLoader[Any],
    ) -> None:
        """Initialize object of class Evaluator

        Args:
            model: The model evaluating our data-set
            pipeline: The development's pipeline used to pre-process the data-set
            data_loader: The data loader used to efficiently load the data_set
        """
        self.model = model
        self.pipeline = pipeline
        self.data_loader = data_loader

        text_config = TextConfig()
        image_config = ImageConfig()
        audio_config = AudioConfig()

        self.pipeline_dict: PipelineMapping = {
            "text": TextPipeline(text_config),
            "image": ImagePipeline(image_config),
            "audio": AudioPipeline(audio_config),
        }

        self.multimodal_pipeline = MultimodalPipeline(self.pipeline_dict)

    def compute(
        self, data: Dict[ImageT | TextT | AudioT, List[ImageT | TextT | AudioT]]
    ) -> List[torch.Tensor]:
        """
        Evaluate the model on a dictionary of modality data.

        Args:
            data: Dictionary where keys are modalities ('image', 'text', 'audio')
                and values are lists of data for that modality

        Returns:
            List of model outputs
        """
        self.model.eval()
        predictions: List[torch.Tensor] = []

        with torch.no_grad():
            for modality, raw_data in data.items():
                if modality == "text" and isinstance(raw_data, str):
                    inputs = self.pipeline_dict["text"].process(raw_data)
                elif modality == "image" and isinstance(raw_data, torch.Tensor):
                    inputs = self.pipeline_dict["image"].process([raw_data.numpy()])
                elif modality == "audio" and isinstance(raw_data, torch.Tensor):
                    inputs = self.pipeline_dict["audio"].process(raw_data.numpy())
                else:
                    raise ValueError(f"Unsupported modality or data type: {modality}")

                outputs = self.model(inputs)
                predictions.append(outputs)

        return predictions
