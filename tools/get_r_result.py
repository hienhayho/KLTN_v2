import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--thr", type=float, default=0.7, help="Threshold for evaluation metrics"
)
parser.add_argument("--input-dir", type=str, required=True, help="Input directory path")
args = parser.parse_args()

subdirs = [f for f in Path(args.input_dir).iterdir() if f.is_dir()]
if len(subdirs) == 0:
    subdirs = [Path(args.input_dir)]

for subdir in subdirs:
    metric = subdir.name

    txt_files = subdir.glob("*.txt")

    scores = []
    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            scores.append(float(f.read().strip()))

    passed = [1 if score >= args.thr else 0 for score in scores]
    avg_score = sum(passed) / len(scores)
    print(f"{metric}: {avg_score:.2%}")
