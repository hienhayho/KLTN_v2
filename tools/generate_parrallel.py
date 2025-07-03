import json
import argparse
from tqdm import tqdm
from pathlib import Path
from json_repair import loads
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingMode,
    OpenAIEmbeddingModelType,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings

from src.retrieve import retrieve_sync

load_dotenv()

# - Each question must explicitly mention the name of the document.

# Prompt definitions
system_message = """You are an expert question generator. Your task is to create questions based on the provided document content.

**Instruction**:
- You will be given a passage that includes the document name and a portion of its content.
- Generate up to 5 questions that can be answered solely from the provided content.
- Your questions must be in Vietnamese.
- Return only the list of questions in the following JSON format:
```json
{
    "questions": ["Question 1", "Question 2", "..."]
}
```
"""

reasoning_prompt = """You are a reasoning assistant. Your task is to answer the given question using the provided context.

**Instruction**:

- First, provide a step-by-step reasoning about how to answer the question based on the context in Vietnamese.
- Make sure that only the quoted parts are copied verbatim from the context; the rest of the reasoning must be in your own words.
- Conclude your response with the final answer using the format: <ANSWER>: final_answer.
 You MUST start the final answer line with the tag "<ANSWER>:" exactly.

**Important**:

- Be thorough and logical in the reasoning. Do not include any other formatting or explanation outside the instructions."""


llm = OpenAI(model="gpt-4o-mini")
embed_model = OpenAIEmbedding(
    mode=OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
    model=OpenAIEmbeddingModelType.TEXT_EMBED_3_LARGE,
)
Settings.embed_model = embed_model


def process_file(txt_file_path: str, output_dir: Path):
    txt_file = Path(txt_file_path)
    doc_id = txt_file.stem
    results = []

    try:
        with open(txt_file, "r") as f:
            text = f.read()

        messages = [
            ChatMessage(role="system", content=system_message),
            ChatMessage(role="user", content="Passage content: " + text),
        ]

        output = llm.chat(
            messages=messages, response_format={"type": "json_object"}
        ).message.content
        result_output = loads(output)
        questions = result_output.get("questions", [])

        print(f"[{doc_id}] len(questions): {len(questions)}")

        for q in questions:
            contexts = retrieve_sync(q, top_k=8, method="hybrid")

            cot_answer = llm.chat(
                [
                    ChatMessage(role="system", content=reasoning_prompt),
                    ChatMessage(
                        role="user",
                        content=f"Context: {text}\n==================== \n Question: {q}",
                    ),
                ]
            ).message.content

            print("question:", q)
            print("===============================")
            print("cot_answer:", cot_answer)
            print()

            results.append(
                {
                    "doc_id": doc_id,
                    "true_context": text,
                    "question": q,
                    "cot_answer": cot_answer,
                    "contexts": contexts,
                }
            )

        # Save to individual file
        out_path = output_dir / f"{doc_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"[ERROR] {txt_file}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    txt_files = list(Path(args.path).rglob("*.txt"))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(txt_files)} .txt files")
    print(f"Saving individual results to: {output_dir.resolve()}")

    tasks = []
    for file in txt_files:
        doc_id = Path(file).stem
        output_file = output_dir / f"{doc_id}.json"
        if output_file.exists():
            print(f"[SKIP] {doc_id}.json already exists.")
            continue
        tasks.append(file)

    print(f"Processing {len(tasks)} files (skipped {len(txt_files) - len(tasks)})")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_file, file, output_dir): file for file in tasks
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Processing(num_proc={args.workers}) ...",
        ):
            _ = future.result()

    print("[DONE] All new documents processed and saved.")


if __name__ == "__main__":
    main()
