import chromadb
from loguru import logger
from typing import Literal
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo

from src.utils import config

load_dotenv()

embed_model = OpenAIEmbedding(model=config.embed_model_name)
Settings.embed_model = embed_model


def load_semantic_retriver():
    vector_db_collection = config.vector_db_collection
    db2 = chromadb.PersistentClient(path=vector_db_collection)

    # chroma_collection = db2.get_or_create_collection("quickstart")
    chroma_collection = db2.get_or_create_collection(vector_db_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    vector_store_info = VectorStoreInfo(
        content_info="Metadata of document about public administrative services",
        metadata_info=[
            MetadataInfo(
                name="file_name",
                type="str",
                description=(
                    "Name of the file containing the document about public "
                    "administrative services"
                ),
            ),
            MetadataInfo(
                name="law_name",
                type="str",
                description=(
                    "Name of the law containing the document about public "
                    "administrative services"
                ),
            ),
            MetadataInfo(
                name="doc_name",
                type="str",
                description="Name of the document about public administrative services",
            ),
        ],
    )
    retriever = VectorIndexAutoRetriever(
        index, vector_store_info=vector_store_info, similarity_top_k=8
    )

    logger.info("Loaded semantic retriever")
    return retriever


semantic_retriever = load_semantic_retriver()


def load_bm25_retriever():
    bm_25_index = config.bm_25_index
    loaded_bm25_retriever = BM25Retriever.from_persist_dir(bm_25_index)

    logger.info("Loaded BM25 retriever")
    return loaded_bm25_retriever


bm25_retriver = load_bm25_retriever()


def load_hybrid_search(weights: list[float] | None = None):

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[semantic_retriever, bm25_retriver],
        num_queries=1,
        similarity_top_k=8,
        use_async=True,
        retriever_weights=weights,
        mode="relative_score",
    )
    logger.info("Loaded hybrid retriever")

    return hybrid_retriever


hybrid_retriever = load_hybrid_search(
    weights=[
        config.semantic_retriever_weight,
        config.bm25_retriever_weight,
    ]
)


async def retrive(
    query: str, top_k: int, method: Literal["semantic", "bm25", "hybrid"] = "hybrid"
) -> list[str]:
    """
    Retrieve documents based on the input query using a hybrid retriever.

    Args:
        query (str): The input query.
        top_k (int): The number of top documents to retrieve.
        method (Literal["semantic", "bm25", "hybrid"]): The retriever method to use.
            - "semantic": Use semantic retrieval.
            - "bm25": Use BM25 retrieval.
            - "hybrid": Use hybrid retrieval (combination of semantic and BM25).

    Returns:
        list: A list of retrieved documents.
    """
    assert method in [
        "semantic",
        "bm25",
        "hybrid",
    ], "Invalid retriever method. Use 'semantic', 'bm25', or 'hybrid'."

    if method == "semantic":
        retriever = semantic_retriever
    elif method == "bm25":
        retriever = bm25_retriver
    else:
        retriever = hybrid_retriever

    documents = await retriever.aretrieve(query)

    contexts = []
    for node in documents[:top_k]:
        contexts.append(node.text)

    return contexts
