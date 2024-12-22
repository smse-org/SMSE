import torch
from torch.utils.data import DataLoader
from typing import TypeVar, Union

from smse.pipelines.text import TextPipeline
from smse.pipelines.image import ImagePipeline
from smse.pipelines.audio import AudioPipeline
from smse.pipelines.factory import MultimodalPipeline

PipelineDict = Union[TextPipeline, ImagePipeline, AudioPipeline, MultimodalPipeline]
Pipeline = TypeVar('T', TextPipeline, ImagePipeline, AudioPipeline, MultimodalPipeline)

class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        pipeline: Pipeline,
        data_loader: DataLoader,
    ) -> None:
        self.model: torch.nn.Module = model
        self.pipeline: Pipeline = pipeline
        self.data_loader: DataLoader = data_loader

    def _select_pipeline(self, modality):
        """Select pipeline based on modality."""
        if modality not in PipelineDict:
            raise ValueError(f"No pipeline found for modality: {modality}")
        return self.pipeline_dict[modality]

    def compute(self, modality):
        """
        Evaluate the model on a specific modality.

        :param modality: The modality to evaluate ('image',
                                                    'text',
                                                    'audio',
                                                    'multimodal').
        :return: List of model outputs.
        """
        pipeline = self._select_pipeline(modality)
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for raw_data in self.data_loader:
                inputs = pipeline.process(raw_data)
                outputs = self.model(inputs)
                predictions.append(outputs)

        return predictions
