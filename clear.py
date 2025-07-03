import polars as pl
from tqdm import tqdm

pl.Config.set_tbl_rows(10000)
pl.Config.set_tbl_cols(1000)
pl.Config.set_fmt_str_lengths(200)

data = pl.read_ndjson("data_sft_think_gemini_20k_SW_eq_9k6_tool.jsonl")

results = []
for row in tqdm(data.iter_rows(named=True), total=data.height):
    r = {}
    r["tools"] = row["tools"]
    r["messages"] = []
    for m in row["messages"]:
        if "annotations" in m:
            del m["annotations"]
        r["messages"].append(m)
    results.append(r)

print(len(results))
pl.DataFrame(results).write_ndjson("processed_data_full.jsonl")

new_data = pl.read_ndjson("processed_data_full.jsonl")
new_data[0]
