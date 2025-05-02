import argparse
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()


def generate_summary(path: str):
    # Read content from the file
    with open(path, "r") as file:
        content = file.read()

    llm = OpenAI(model="gpt-4.1-mini")

    system_prompt = """Bạn là người am hiểu tiếng việt, nhiệm vụ của bạn là tóm tắt nội dung đoạn văn được cung cấp trong một câu.

    ### Hướng dẫn
    - Hãy đề cập đầy đủ tất cả nội dung chính của đoạn văn được cung cấp.
    - Trong câu tóm tắt của bạn, hãy đề cập đến đoạn văn được cung cấp nằm trong tài liệu nào.
    - Không cần thêm phần thừa thãi như: "Đây là tóm tắt của đoạn văn này", "Đoạn văn này nói về", ...

    Tôi sẽ thưởng cho bạn 10000 điểm nếu bạn làm tốt nhiệm vụ này.
    """

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(
            role="user", content=f"""Đoạn văn được cung cấp như sau\n: {content}"""
        ),
    ]

    response = llm.chat(messages)

    return response.message.content.replace(".docx", ""), content.replace(".docx", "")


def main():
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [str(file) for file in Path(args.data).glob("*.txt")]

    for file in tqdm(files):
        summary, content = generate_summary(file)

        new_content = f"""{summary}\n\n{content}"""
        with open(out_dir / Path(file).name, "w") as f:
            f.write(new_content)


if __name__ == "__main__":
    main()
