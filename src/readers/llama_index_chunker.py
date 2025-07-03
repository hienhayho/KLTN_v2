from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Optional, List
from llama_index.core import Document, Settings
from llama_index.embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingMode,
    OpenAIEmbeddingModelType,
)
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    TokenTextSplitter,
    SentenceSplitter,
)

load_dotenv()

embed_model = OpenAIEmbedding(
    mode=OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
    model=OpenAIEmbeddingModelType.TEXT_EMBED_3_LARGE,
)
Settings.embed_model = embed_model


class ChunkerBasedReader(BaseReader):
    def __init__(self, method: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if method == "semantic":
            self.node_parser = SemanticSplitterNodeParser(
                embed_model=embed_model,
                buffer_size=1,
                breakpoint_percentile_threshold=90,
            )
        elif method == "token":
            self.node_parser = TokenTextSplitter(
                chunk_size=512,
                chunk_overlap=20,
            )
        elif method == "sentence":
            self.node_parser = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=20,
            )

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs=None,
    ) -> List[Document]:
        """Parse file."""
        if not isinstance(file, Path):
            file = Path(file)

        try:
            import docx2txt
        except ImportError:
            raise ImportError(
                "docx2txt is required to read Microsoft Word files: "
                "`pip install docx2txt`"
            )

        if fs:
            with fs.open(str(file)) as f:
                text = docx2txt.process(f)
        else:
            text = docx2txt.process(file)
        metadata = {
            "file_name": file.name,
            "law_name": file.parent.name,
            "doc_name": file.name,
        }
        if extra_info is not None:
            metadata.update(extra_info)

        idx = text.find("1. mục đích")
        text = text[idx:] if idx != -1 else text

        # Split the text into chunks using the SemanticSplitterNodeParser
        documents = self.node_parser.get_nodes_from_documents(
            documents=[Document(text=text, metadata=metadata)]
        )

        chunks: list[Document] = []
        for doc in documents:
            text = doc.text
            metadata = doc.metadata

            text = f"Bộ luật: {file.parent.name} - Tài liệu: {file.stem}\n" + text
            chunks.append(
                Document(
                    text=text,
                    metadata=metadata,
                )
            )

        return chunks
