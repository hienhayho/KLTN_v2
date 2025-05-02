import chromadb
import argparse
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from llama_index.embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingMode,
    OpenAIEmbeddingModelType,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

# from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    StorageContext,
)

load_dotenv()

parser = argparse.ArgumentParser(description="Ingest documents")
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
args = parser.parse_args()

embed_model = OpenAIEmbedding(
    mode=OpenAIEmbeddingMode.SIMILARITY_MODE,
    model=OpenAIEmbeddingModelType.TEXT_EMBED_3_LARGE,
)
Settings.embed_model = embed_model

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

db = chromadb.PersistentClient(path=str(db_out_dir))

chroma_collection = db.get_or_create_collection(args.o)

vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection,
    collection_kwargs={"metadata": {"hnsw:space": "cosine"}},
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
