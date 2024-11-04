from dataclasses import dataclass
from typing import Optional, Union, List, Any
from pathlib import Path
from pipelines.base_pipeline import PipelineConfig, Pipeline


@dataclass
class TextConfig(PipelineConfig):
    chunk_overlap: int = 0
    tokenizer: Optional[Any] = None
    max_sequence_length: int = 512


class TextPipeline(Pipeline):
    def __init__(self, config: TextConfig):
        super().__init__(config)
        self.config: TextConfig = config
        self._tokenizer = config.tokenizer

    def load(self, input_path: Union[str, Path]) -> str:
        """Load text from file"""
        with open(input_path, "r", encoding="utf-8") as f:
            return f.read()

    def validate(self, data: Any) -> bool:
        return isinstance(data, str)

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks based on max_sequence_length"""
        if not self.config.max_sequence_length:
            return [text]

        # Simple chunking strategy - can be extended with more sophisticated approaches
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > self.config.max_sequence_length:
                if current_chunk:  # Avoid empty chunks
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def preprocess(self, text: str) -> Union[List[str], List[List[int]]]:
        """Preprocess text data"""
        chunks = self._split_into_chunks(text)

        if self._tokenizer:
            return [self._tokenizer(chunk) for chunk in chunks]
        return chunks
