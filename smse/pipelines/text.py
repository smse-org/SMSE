from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import torch
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,  # type: ignore[import-untyped,import-not-found]
)
from torch import Tensor

from smse.pipelines.base import BaseConfig, BasePipeline
from smse.types import TextT


@dataclass
class TextConfig(BaseConfig):
    """Configuration class for text pipeline"""

    chunk_size: int = 512
    """Maximum sequence length for tokenization"""

    chunk_overlap: int = 0
    """Number of overlapping characters between chunks"""

    tokenizer: Optional[Any] = None
    """Callable Tokenizer object to use for tokenization"""


class TextPipeline(BasePipeline):
    def __init__(self, config: TextConfig):
        super().__init__(config)
        self.config: TextConfig = config
        self._tokenizer = config.tokenizer

    def load(self, input_paths: List[Path]) -> List[TextT]:
        """Load text from a list of files"""
        texts = []
        for input_path in input_paths:
            with open(input_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        return texts

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

    def process(self, texts: List[TextT]) -> List[List[TextT]] | Tensor:
        """
        Process a batch of texts.

        Args:
            texts (List[TextT]): List of text data.

        Returns:
            List[List[TextT]] | Tensor: Processed text data. If tokenizer is
                provided, returns tokenized text data. [n, chunks, text/tokens]
        """
        processed_texts: List[Any] = []

        for text in texts:
            chunks = self._split_into_chunks(text)

            if self._tokenizer:
                processed_texts.append(
                    torch.stack([self._tokenizer(chunk) for chunk in chunks])
                )
            else:
                processed_texts.append(chunks)

        if self._tokenizer:
            return torch.concat(processed_texts).to(self.config.device)
        else:
            return processed_texts
