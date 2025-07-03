import json
import argparse
from tqdm import tqdm
from pathlib import Path
from json_repair import loads
from dotenv import load_dotenv
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

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

llm = OpenAI(model="gpt-4.1-mini")
embed_model = OpenAIEmbedding(
    mode=OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
    model=OpenAIEmbeddingModelType.TEXT_EMBED_3_LARGE,
    # model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
)
Settings.embed_model = embed_model

system_message = """You are an expert question generator. Your task is to create questions based on the provided document content.

**Instruction**:
- You will be given a passage that includes the document name and a portion of its content.
- Generate up to 5 questions that can be answered solely from the provided content.
- Each question must explicitly mention the name of the document.
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
- Conclude your response with the final answer using the format: <ANSWER>: {{final_answer}}.
- You MUST start the final answer line with the tag "<ANSWER>:" exactly.

**Important**:
- Be thorough and logical in the reasoning.
- Do not include any other formatting or explanation outside the instructions."""

txt_files = list(Path(args.path).rglob("*.txt"))

# print(list(txt_files))

results = []

for txt_file in tqdm(txt_files, total=len(txt_files)):
    doc_id = txt_file.stem
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

    print("len(questions)", len(result_output["questions"]))

    for q in result_output["questions"]:
        contexts = retrieve_sync(q, top_k=8, method="hybrid")

        result = dict()

        cot_answer = llm.chat(
            [
                ChatMessage(role="system", content=reasoning_prompt),
                ChatMessage(
                    role="user",
                    content=f"Context: {text}\n==================== \n Question: {q}",
                ),
            ],
        ).message.content

        result["doc_id"] = doc_id
        result["true_context"] = text
        result["question"] = q
        result["cot_answer"] = cot_answer
        result["contexts"] = contexts

        print("question: ", q)
        print("=====================")
        print("true_context: ", text)
        print("======================")
        print("cot_answer: ", cot_answer)
        print("======================")
        # result = {
        #     "doc_id": doc_id,
        #     "context": text,
        #     "question": q,
        #     "cot_answer": cot_answer,
        #     "contexts": contexts,
        # }

        results.append(result)


with open(args.output, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
