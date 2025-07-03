import argparse
import polars as pl
from pathlib import Path

from src.prompt import get_prompt, gen_data_prompt

parser = argparse.ArgumentParser()
parser.add_argument("--max-ref-docs", type=int, default=5)
parser.add_argument("--not-cot", action="store_true")
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument(
    "--mode",
    type=str,
    choices=["json", "plain_text", "markdown", "yaml", "xml"],
    required=True,
)
args = parser.parse_args()


def read_json_files(data_dir):
    data = []
    for file in Path(data_dir).glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data.extend(pl.read_json(f).to_dicts())
    return data


data = read_json_files(args.data_dir)

results = []
counts = 0


system_prompt, user_prompt_template = get_prompt(
    prompt=gen_data_prompt, mode=args.mode, is_cot=not args.not_cot
)

print(system_prompt)
print("===========================")
print(user_prompt_template)

for row in data:
    true_context = row["true_context"]
    question = row["question"]
    cot_answer = row["cot_answer"]
    contexts = row["contexts"]

    if args.not_cot:
        cot_answer = cot_answer.split("<ANSWER>:")[-1].strip()

    filtered_contexts = contexts[: args.max_ref_docs]

    final_context = ""
    for c in filtered_contexts:
        final_context += "<DOCUMENT>" + c + "</DOCUMENT>\n"

    user_prompt = user_prompt_template.format(query=question, context=final_context)
    answer = f"<REASON>: {cot_answer}"

    # results.append(
    #     {
    #         "conversations": [
    #             {"from": "human", "value": user_prompt},
    #             {"from": "gpt", "value": answer},
    #         ],
    #         "system": "",
    #         "tools": "",
    #     }
    # )
    results.append(
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": answer},
            ]
        }
    )

print(f"Total samples: {len(results)}")

pl.DataFrame(results).write_ndjson(args.output)
