from pathlib import Path
from typing import TypeVar, Union, Any, Dict

from smse.pipelines.text import TextPipeline
from smse.pipelines.image import ImagePipeline
from smse.pipelines.audio import AudioPipeline

Pipeline = TypeVar('T', TextPipeline, ImagePipeline, AudioPipeline)

class MultimodalPipeline:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def _process_single(self, data_type: str, input_data: Any) -> Any:
        """
        Preprocess data for a single modality.
        :param data_type: Modality type (e.g., 'image', 'text').
        :param input_data: Raw input data for the modality.
        :return: Processed data.
        """
        if data_type not in self.pipeline:
            raise ValueError(f"No pipeline configured for data type: {data_type}")
        return self.pipeline[data_type].process(input_data)

    def process(self, inputs: Dict[str, Union[str, Path, Any]]) -> Dict[str, Any]:
        """Process multiple modalities"""
        return {data_type: self._process_single(data_type, input_data) for data_type, input_data in inputs.items()}
