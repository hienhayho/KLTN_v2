import json
import time
import asyncio
import argparse
import polars as pl
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Literal
from mmengine.config import DictAction
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

from src.utils import config
from src.flow import AppFlow
from src.prompt import llm_as_a_judge_prompt, get_prompt


llm = OpenAI(model="gpt-4.1-mini", temperature=0.0)


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--i", type=str, help="Input CSV file")
    parser.add_argument("--o", type=str, required=True, help="Output dir")
    parser.add_argument(
        "--only-retrieve",
        action="store_true",
        help="Only retrive the context, do not run the app flow",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    return parser.parse_args()


async def llm_as_a_judge(
    query: str,
    ground_truth: str,
    answer: str,
    mode: Literal["plain_text", "json"] = "plain_text",
):
    """
    Evaluate the answer using LLM as a judge.
    """
    system_prompt, user_prompt = get_prompt(prompt=llm_as_a_judge_prompt, mode=mode)
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(
            role="user",
            content=user_prompt.format(
                query=query, ground_truth=ground_truth, answer=answer
            ),
        ),
    ]

    response = await llm.achat(messages, response_format={"type": "json_object"})
    result = json.loads(response.message.content)
    return result["is_correct"]


async def main(args):
    data = pl.read_csv(args.i)
    logger.info(f"Loaded {len(data)} questions ...")

    Path(args.o).mkdir(parents=True, exist_ok=True)
    config.dump(Path(args.o) / "config.py")

    bar = tqdm(data.iter_rows(named=True), total=len(data), desc="Evaluating ...")

    correct = 0
    num_questions = 0
    total_times = 0
    all_pairs = []

    for d in bar:
        query = d["Câu hỏi"]
        stt = d["STT"]
        ground_truth = d["Groundtruth"]
        expected_answer = d["Answer"]

        start_time = time.time()

        app_flow = AppFlow(timeout=1000, verbose=False)
        result = await app_flow.run(
            query=query, history=[], only_retrieve=args.only_retrieve
        )
        end_time = time.time()

        if ground_truth and not args.only_retrieve:
            is_correct = await llm_as_a_judge(
                query=query,
                ground_truth=ground_truth,
                answer=result.answer,
            )
        else:
            # Placeholder for no ground truth
            is_correct = True

        all_pairs.append(
            {
                "stt": stt,
                "query": query,
                "ground_truth": ground_truth,
                "contexts": result.contexts,
                "final_query": result.final_query,
                "expected_answer": expected_answer,
                "answer": result.answer,
                "is_correct": is_correct,
            }
        )

        correct += is_correct
        num_questions += 1
        total_times += end_time - start_time

        bar.set_postfix(
            {
                "accuracy": f"{correct / num_questions:.4f}",
                "time_average": f"{total_times / num_questions:.4f}s",
            }
        )

        await asyncio.sleep(0.5)

    final_result = {
        "accuracy": correct / num_questions,
        "total_time": total_times,
        "time_average": total_times / num_questions,
        "num_questions": num_questions,
    }

    with open(Path(args.o) / "eval_acc_result.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4)

    pl.DataFrame(all_pairs).write_ndjson(Path(args.o) / "all_pairs.jsonl")

    logger.info(f"Saved result to {args.o}")
    logger.info(f"Accuracy: {correct / num_questions:.4f}")
    logger.info(f"Total time: {total_times:.2f}s")
    logger.info(f"Average time: {total_times / num_questions:.2f}s")


if __name__ == "__main__":
    args = get_args()
    if args.cfg_options is not None:
        config.merge_from_dict(args.cfg_options)

    print(config.pretty_text)
    input("Press Enter to continue...")
    asyncio.run(main(args))
