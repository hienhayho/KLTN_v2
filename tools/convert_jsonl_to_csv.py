import argparse
import polars as pl
from pathlib import Path

parser = argparse.ArgumentParser(description="Convert JSONL files to CSV format.")
parser.add_argument(
    "--i",
    type=str,
    help="Path to the input JSONL file.",
)
parser.add_argument(
    "--o",
    type=str,
    help="Path to the output CSV file.",
)
args = parser.parse_args()


def convert_jsonl_to_csv(input_file: str, output_file: str) -> None:
    """
    Convert a JSONL file to CSV format.

    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to the output CSV file.
    """
    # Read the JSONL file
    df = pl.read_ndjson(input_file)

    results = []
    for row in df.iter_rows(named=True):
        results.append(
            {
                "stt": row["stt"],
                "query": row["query"],
                "ground_truth": row["ground_truth"],
                "bot_answer": row["answer"],
                "bot_judge": row["is_correct"],
                "user_judge": row["is_correct"],
            }
        )

    # Create a DataFrame from the results
    df = pl.DataFrame(results)
    # Write the DataFrame to a CSV file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_file)


if __name__ == "__main__":
    convert_jsonl_to_csv(args.i, args.o)
