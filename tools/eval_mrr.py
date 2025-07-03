import json
import argparse
import polars as pl
from tqdm import tqdm
from pathlib import Path
from json_repair import loads
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

load_dotenv()

llm = OpenAI(model="gpt-4.1-mini", temperature=0.0)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to the input JSONL file containing the evaluation data.",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path to the output json file to save the MRR results.",
)
parser.add_argument(
    "--top-k",
    type=int,
    nargs="+",
    default=[1, 5],
    help="Number of documents to consider for the evaluation.",
)
args = parser.parse_args()

system_prompt = """Persona
- You are a helpful assistant that can decide the provided chunk is relevant or not with the query.

Instructions
- Given a query and a chunk of text, determine if the chunk is relevant to the query.
- A relevant chunk is one that contains information that is useful and enough for directly answering the query.

Format
- Output must be in JSON format.
{{
    "is_relevant": <boolean>,
}}

Notes
- You will be rewarded $1000 if you finish the task accurately.
"""

user_prompt = """Chunk
{chunk}

Query
{query}"""


def get_rr(rank: int) -> float:
    """
    Calculate reciprocal rank (RR) given the rank of the first relevant document.
    """
    if rank == 0:
        return 0.0
    return 1.0 / rank


async def check_relevant(prompt: str) -> int:
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=prompt),
    ]
    response = await llm.achat(
        messages=messages, response_format={"type": "json_object"}
    )
    content = loads(response.message.content)
    return content["is_relevant"]


async def get_single_rr(question: str, contexts: list[str], k: int) -> float:
    """
    Calculate reciprocal rank (RR) for a single query and its contexts.
    """
    for i, context in enumerate(contexts[:k]):
        prompt = user_prompt.format(chunk=context, query=question)
        is_relevant = await check_relevant(prompt)
        if is_relevant:
            return get_rr(i + 1)

    return 0.0


async def get_mrr(df: pl.DataFrame, ks: list[int]) -> dict:
    mrr_results = {k: 0.0 for k in ks}
    results = dict(mrr=dict(), data=[])
    count = 0
    bar = tqdm(df.iter_rows(named=True), desc="Calculating MRR ...", total=len(df))
    for row in bar:
        if not row["ground_truth"]:
            continue

        count += 1
        question = row["query"]
        contexts = row["contexts"]
        stt = row["stt"]
        rrs = await asyncio.gather(*[get_single_rr(question, contexts, k) for k in ks])

        for i, k in enumerate(ks):
            mrr_results[k] += rrs[i]

        results["data"].append(
            {
                "query": question,
                "ground_truth": row["ground_truth"],
                "stt": stt,
                "contexts": contexts,
                "mrr": {k: rrs[i] for i, k in enumerate(ks)},
            }
        )
        bar.set_postfix({f"mrr@{k}": round(mrr_results[k] / count, 4) for k in ks})

    for k in ks:
        mrr_results[k] /= count
        mrr_results[k] = round(mrr_results[k], 4)

    bar.close()
    results["mrr"] = mrr_results
    return results


async def main():
    df = pl.read_ndjson(args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    result = await get_mrr(df, args.top_k)

    file_name = f"{Path(args.output).stem}"
    for k in result["mrr"].keys():
        file_name += f"_mrr@{k}-{result['mrr'][k]}"

    file_name += ".json"
    args.output = Path(args.output).parent / file_name
    with open(args.output, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
