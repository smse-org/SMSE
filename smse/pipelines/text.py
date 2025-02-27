from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

from smse.pipelines.base import BaseConfig, BasePipeline
from smse.types import TextT


@dataclass
class TextConfig(BaseConfig):
    """Configuration class for text pipeline"""

    chunk_overlap: int = 0
    """Number of overlapping tokens between chunks"""

    tokenizer: Optional[Any] = None
    """Tokenizer object to use for tokenization"""

    max_sequence_length: int = 512
    """Maximum sequence length for tokenization"""


class TextPipeline(BasePipeline):
    def __init__(self, config: TextConfig):
        super().__init__(config)
        self.config: TextConfig = config
        self._tokenizer = config.tokenizer

    def load(self, input_path: Union[TextT, Path]) -> TextT:
        """Load text from file"""
        with open(input_path, "r", encoding="utf-8") as f:
            return f.read()

    def validate(self, data: Any) -> bool:
        return isinstance(data, TextT)

    def _split_into_chunks(self, text: TextT) -> List[TextT]:
        """Split text into chunks based on max_sequence_length"""
        if not self.config.max_sequence_length:
            return [text]

        # Simple chunking strategy - can be extended with more sophisticated approaches
        words = text.split()
        chunks = []
        current_chunk: List[str] = []
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

    def process(self, text: TextT) -> Union[List[TextT], List[List[int]]]:
        """Preprocess text data"""
        chunks = self._split_into_chunks(text)

        if self._tokenizer:
            return [self._tokenizer(chunk) for chunk in chunks]
        return chunks
