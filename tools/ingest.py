import chromadb
import argparse
import uuid
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingMode,
    OpenAIEmbeddingModelType,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    StorageContext,
)

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest documents")
    parser.add_argument(
        "--root",
        type=str,
        required=False,
        # default="data/Quy_trinh_ISO_Word",
        help="Root directory containing raw documents.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["semantic", "custom", "token", "sentence"],
        default="custom",
    )
    parser.add_argument(
        "--i",
        type=str,
        default="./data",
        help="Directory containing the documents to ingest",
    )
    parser.add_argument(
        "--o",
        type=str,
        default="./output",
        help="Directory to save the output index",
    )
    return parser.parse_args()


embed_model = OpenAIEmbedding(
    mode=OpenAIEmbeddingMode.SIMILARITY_MODE,
    model=OpenAIEmbeddingModelType.TEXT_EMBED_3_LARGE,
)
Settings.embed_model = embed_model


def load_reader(args):
    logger.info(f"Reader: {args.method}")
    if args.method == "custom":
        from src.readers.rule_based_chunker import CustomReader

        return CustomReader(
            titles=[
                "1. mục đích",
                "2. phạm vi áp dụng",
                "3. tài liệu viện dẫn",
                "4. định nghĩa/viết tắt",
                "5. nội dung quy trình",
                "6. biểu mẫu",
                ". hồ sơ lưu",
            ]
        )
    else:
        from src.readers.llama_index_chunker import ChunkerBasedReader

        return ChunkerBasedReader(
            method=args.method,
        )


def get_chunks(args):
    logger.info(f"Loading documents from {args.root}")
    reader = load_reader(args)
    data = SimpleDirectoryReader(
        args.root,
        recursive=True,
        file_extractor={".docx": reader},
    ).load_data(show_progress=True)

    return data


def save_chunks(args, data):
    logger.info(f"Saving chunks to {args.i}")
    output_folder = Path(args.i)
    output_folder.mkdir(exist_ok=True, parents=True)

    for d in tqdm(data):
        with open(output_folder / f"{d.doc_id}.txt", "w") as f:
            f.write(d.text)


def ingest_documents(args):
    logger.info("Ingesting documents...")

    if args.root:
        chunks = get_chunks(args)
        save_chunks(args, chunks)

    output_dir = Path(args.o)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_out_dir = Path(args.o) / "chroma_db"
    db_out_dir.mkdir(parents=True, exist_ok=True)

    data = []
    files = list(Path(args.i).rglob("*.txt"))
    for file in tqdm(files, desc="Loading files"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            data.append(Document(text=content))

    client = chromadb.PersistentClient(
        path=str(db_out_dir),
    )

    collection = client.get_or_create_collection(
        name=(Path(db_out_dir).parent.name),
        configuration={"hnsw": {"space": "cosine"}},
    )

    vector_store = ChromaVectorStore(
        chroma_collection=collection,
        persist_dir=str(db_out_dir),
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(data, storage_context=storage_context, show_progress=True)

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=data,
        similarity_top_k=8,
    )

    bm25_out_dir = Path(args.o) / "bm25_retriever"
    bm25_retriever.persist(str(bm25_out_dir))

    test_result = bm25_retriever.retrieve("xin điều kiện kết hôn cho nam")

    for node in test_result:
        print(node.text)
        print("\n==========\n\n")


if __name__ == "__main__":
    args = parse_args()
    ingest_documents(args)
