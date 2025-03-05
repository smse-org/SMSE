from pathlib import Path
from typing import Any, Dict, Mapping, Union

from smse.pipelines.audio import AudioPipeline
from smse.pipelines.image import ImagePipeline
from smse.pipelines.text import TextPipeline

# Create a mapping type for the pipeline dictionary
PipelineMapping = Mapping[str, Union[TextPipeline, ImagePipeline, AudioPipeline]]


class MultimodalPipeline:
    def __init__(self, pipelines: PipelineMapping):
        """
        Initialize MultimodalPipeline with a mapping of modality types to
        their respective pipelines.

        Args:
            pipelines: A mapping from modality names to their pipeline instances
        """
        self.pipelines = pipelines

    def _process_single(self, data_type: str, input_data: Any) -> Any:
        """
        Preprocess data for a single modality.

        Args:
            data_type: Modality type (e.g., 'image', 'text')
            input_data: Raw input data for the modality

        Returns:
            Processed data

        Raises:
            ValueError: If no pipeline is configured for the data type
        """
        if data_type not in self.pipelines:
            raise ValueError(f"No pipeline configured for data type: {data_type}")
        return self.pipelines[data_type].process(input_data)

    def process(self, inputs: Dict[str, Union[str, Path, Any]]) -> Dict[str, Any]:
        """
        Process multiple modalities

        Args:
            inputs: Dictionary mapping modality types to their input data

        Returns:
            Dictionary mapping modality types to their processed data
        """
        return {
            data_type: self._process_single(data_type, input_data)
            for data_type, input_data in inputs.items()
        }
