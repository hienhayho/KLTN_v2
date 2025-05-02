import argparse
import polars as pl
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Benchmark VLLM model.")
parser.add_argument(
    "--model",
    type=str,
    help="Path to the VLLM model.",
)
parser.add_argument(
    "--data",
    type=str,
    help="Path to the test data file.",
)
parser.add_argument(
    "--o",
    type=str,
    help="Path to the output file.",
    required=True,
)
parser.add_argument("--tp", type=int, default=1)
parser.add_argument(
    "--gpu-memory-utilization",
    type=float,
    default=0.9,
)
args = parser.parse_args()


def load_llm():
    return LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tp,
    )


user_prompt = """
Bạn có khả năng trả lời các câu hỏi multiple choices. Bạn sẽ được cung cấp một câu hỏi và các đáp án để bạn lựa chọn. Hãy trả về đáp án đúng nhất theo dạng A, B, C, D hoặc E.

Câu hỏi: {question}
===========
Các đáp án:
{choices}
===========

Câu trả lời của bạn:
"""


def main():
    data = pl.read_ndjson(args.data)
    llm = load_llm()
    guided_decoding_params = GuidedDecodingParams(choice=["A", "B", "C", "D", "E"])
    sampling_params = SamplingParams(
        guided_decoding=guided_decoding_params, temperature=0.0
    )
    results = []

    bar = tqdm(data.iter_rows(named=True), total=len(data), desc="Evaluating ...")

    for d in bar:
        question = d["question"]
        choices = d["choices"]
        id = d["id"]
        choices_str = "\n".join(choices)
        prompt = user_prompt.format(question=question, choices=choices_str)
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        results.append(
            {
                "id": id,
                "answer": response[0].outputs[0].text,
            }
        )

    pl.DataFrame(results).write_csv(args.o)


if __name__ == "__main__":
    main()
