import os
import argparse
import polars as pl
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# from deepeval import evaluate
# from deepeval.evaluate.configs import DisplayConfig, AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    BaseMetric,
)

# from deepeval.evaluate.utils import aggregate_metric_pass_rates

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--i", type=str, required=True, help="Input file path")
parser.add_argument(
    "--output-dir",
    type=str,
    required=True,
    help="Directory to save the output files",
)
parser.add_argument(
    "--ignores",
    type=int,
    nargs="+",
    default=[],
    help="List of IDs to ignore during evaluation",
)
parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model name")
parser.add_argument(
    "--thresholds", type=float, nargs="+", help="Threshold for evaluation metrics"
)
parser.add_argument(
    "--methods",
    type=str,
    nargs="+",
    required=True,
    help="List of methods to evaluate",
    choices=["precision", "recall", "relevancy"],
)
args = parser.parse_args()

if len(args.methods) != len(set(args.methods)):
    raise ValueError("Duplicate methods found in the input list.")


def get_metrics(args, model) -> list[BaseMetric]:
    thresholds = args.thresholds
    methods = args.methods
    assert len(thresholds) == len(
        methods
    ), "Thresholds and methods must match in length."

    metrics = []

    for thr, method in zip(thresholds, methods):
        if method == "precision":
            metrics.append(ContextualPrecisionMetric(threshold=thr, model=model))
        elif method == "recall":
            metrics.append(ContextualRecallMetric(threshold=thr, model=model))
        elif method == "relevancy":
            metrics.append(ContextualRelevancyMetric(threshold=thr, model=model))

    return metrics


def get_small_file():
    data = pl.read_csv(args.i)
    df = data.head(2)
    df.write_csv("small_data.csv")


def build_test_cases(df):
    test_cases = []

    for row in df.iter_rows(named=True):
        if not row["ground_truth"]:
            continue

        test_case = LLMTestCase(
            input=row["query"],
            actual_output=row["answer"],
            expected_output=row["expected_answer"],
            retrieval_context=row["contexts"],
            additional_metadata={
                "row_id": row["stt"],
            },
        )
        test_cases.append(test_case)

    return test_cases


def check_file_content(file_path: str) -> bool:
    with open(file_path, "r") as file:
        content = file.read()
        # Check if the content is empty
        if not content.strip():
            return False
    return True


def eval_each(metric: BaseMetric, threshold: float):
    save_dir = Path(args.output_dir) / metric.__class__.__name__
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n============================\n")
    data = pl.read_ndjson(args.i)
    print("Total test cases:", len(data))

    # Get processed Id
    processed_id = [
        int(file.stem) for file in save_dir.glob("*.txt") if check_file_content(file)
    ]

    processed_id.extend(args.ignores)
    print("Total processed testcases:", len(processed_id))

    scores = []

    if len(processed_id) > 0:
        # Read the scores from the existing files
        for file in save_dir.glob("*.txt"):
            with open(file, "r") as f:
                score = f.read()
                scores.append(float(score))

    df = data.filter(~pl.col("stt").is_in(processed_id))
    print("Remaining:", len(df))

    test_cases = build_test_cases(df)

    for test_case in tqdm(test_cases, desc=f"{metric.__class__.__name__} ..."):
        s = metric.measure(test_case, _show_indicator=False)
        scores.append(s)
        with open(
            save_dir / f"{test_case.additional_metadata['row_id']}.txt", "w"
        ) as f:
            f.write(str(s))

    trues = [s >= threshold for s in scores]
    acc = sum(trues) / len(scores) * 100
    print(f"{metric.__class__.__name__}: {acc:.2f}%, threshold: {threshold:.2f}")
    print("\n============================\n")

    return acc


if __name__ == "__main__":
    # test_cases = build_test_cases(args)
    metrics = get_metrics(args, args.model)

    accs = []
    for metric in metrics:
        acc = eval_each(metric, metric.threshold)
        accs.append(acc)

    with open(Path(args.output_dir) / "accuracy.txt", "w") as f:
        for metric, acc in zip(metrics, accs):
            f.write(f"{metric.__class__.__name__}: {acc:.2f}%\n")

    # results = evaluate(
    #     test_cases=test_cases,
    #     metrics=metrics,
    #     display_config=DisplayConfig(print_results=False),
    #     async_config=AsyncConfig(run_async=False),
    # )

    # # Aggregate the pass rates for each metric
    # aggregate_metric_pass_rates(results.test_results)
