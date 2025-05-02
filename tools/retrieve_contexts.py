import argparse
import polars as pl
from tqdm import tqdm
from pathlib import Path
from mmengine.config import DictAction

from src.flow import AppFlow
from src.utils import logger, config

parser = argparse.ArgumentParser(description="Retrieve contexts")
parser.add_argument("--i", type=str, help="Input CSV file")
parser.add_argument("--i2", type=str, help="Input CSV file 2")
parser.add_argument("--o", type=str, required=True, help="Output jsonl file")
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
args = parser.parse_args()

if args.cfg_options:
    config.merge_from_dict(args.cfg_options)

data = pl.read_csv(args.i)
data_2 = pl.read_csv(args.i2)
logger.info(f"Loaded {len(data)} questions ...")

print(config.pretty_text)
input("Press Enter to continue...")

Path(args.o).parent.mkdir(parents=True, exist_ok=True)


async def main():
    bar = tqdm(
        zip(data.iter_rows(named=True), data_2.iter_rows(named=True)),
        total=len(data),
        desc="Retrieving ...",
    )

    results = []

    for d1, d2 in bar:
        query = d1["Câu hỏi"]
        stt = d1["STT"]
        expected_answer = d1["Answer"]
        ground_truth = d2["ground_truth"]
        answer = d2["answer"]
        is_correct = d2["is_correct"]

        if not ground_truth:
            results.append(
                {
                    "query": query,
                    "final_query": query,
                    "stt": stt,
                    "ground_truth": "",
                    "answer": answer,
                    "expected_answer": expected_answer,
                    "contexts": [],
                    "is_correct": is_correct,
                }
            )

        flow = AppFlow(timeout=1000, verbose=False)
        result = await flow.run(query=query, history=[], only_retrive=True)

        contexts = result.contexts
        final_query = result.final_query

        results.append(
            {
                "query": query,
                "final_query": final_query,
                "stt": stt,
                "ground_truth": ground_truth,
                "answer": answer,
                "expected_answer": expected_answer,
                "contexts": contexts,
                "is_correct": is_correct,
            }
        )

    pl.DataFrame(results).write_ndjson(f"{config.retriever_options.method}_{args.o}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
