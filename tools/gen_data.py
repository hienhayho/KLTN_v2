import polars as pl
from tqdm import tqdm
from datasets import load_dataset

data = load_dataset(
    "BlossomsAI/reduced_vietnamese_instruction_dataset",
    split="train",
    cache_dir="cache_data",
)

results = []
for d in tqdm(data, total=len(data)):
    # print(d)
    r = {
        "instruction": d["instruction"],
        "input": d["input"],
        "output": d["output"],
    }

    results.append(r)

df = pl.DataFrame(results[:1000])

df.write_ndjson("data_small.jsonl")
