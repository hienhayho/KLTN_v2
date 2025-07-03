from pathlib import Path

dir = Path(
    "/mlcv2/WorkingSpace/Personal/hienht/KLTN_new/data_chunks/output_large_hybrid_with_context_copy"
)
output_dir = Path(
    "/mlcv2/WorkingSpace/Personal/hienht/KLTN_new/data_chunks/output_large_hybrid_with_context_copy_modified"
)
output_dir.mkdir(parents=True, exist_ok=True)

files = [file for file in dir.glob("*.txt")]

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        lines = f.read()

    lines = lines.replace(".docx", "")

    file_name = file.name
    with open(output_dir / file_name, "w", encoding="utf-8") as f:
        f.write(lines)
