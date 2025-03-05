from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,  # type: ignore[import-untyped,import-not-found]
)

from smse.pipelines.base import BaseConfig, BasePipeline
from smse.types import TextT


@dataclass
class TextConfig(BaseConfig):
    """Configuration class for text pipeline"""

    chunk_overlap: int = 0
    """Number of overlapping tokens between chunks"""

    tokenizer: Optional[Any] = None
    """Tokenizer object to use for tokenization"""

    chunk_size: int = 512
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
        """Split text into chunks based on chunk_size"""
        if not self.config.chunk_size:
            return [text]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        chunks: List[TextT] = text_splitter.split_text(text)

        return chunks

    def process(self, text: TextT) -> Union[List[TextT], List[List[int]]]:
        """Preprocess text data"""
        chunks = self._split_into_chunks(text)

        if self._tokenizer:
            return [self._tokenizer(chunk) for chunk in chunks]
        return chunks
