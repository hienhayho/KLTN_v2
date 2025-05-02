# from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# print(model)
import polars as pl

data = pl.read_csv("test_564.csv")

df = data.head(5)

df.write_csv("test_564_5.csv")
