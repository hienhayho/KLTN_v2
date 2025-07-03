import argparse
import json
import polars as pl
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--i", type=str, required=True, help="Input file path")
parser.add_argument("--o", type=str, required=True, help="Output file path")
args = parser.parse_args()

# data = json.load(open("data_sft_new_fix_2.json"))
data = pl.read_ndjson(args.i)

results = []
for item in tqdm(data.iter_rows(named=True), total=data.height):
    results.append(item)

with open(args.o, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# df = pl.DataFrame(results)

# df.write_ndjson("data_sft_new_fix_2.jsonl")
