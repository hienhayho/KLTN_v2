import tiktoken
from tqdm import tqdm
from pathlib import Path
from llama_index.readers.file.docs import DocxReader

files = Path(
    "/mlcv2/WorkingSpace/Personal/hienht/KLTN_new/data/Quy_trinh_ISO_Word"
).rglob("*.docx")
files = list([str(file) for file in files])

average_tokens = []
for file in tqdm(files, desc="Processing files", unit="file"):
    # Read the Word document
    content = DocxReader().load_data(file)

    text = content[0].text

    num_words = len(text.split())
    average_tokens.append(num_words)

    # Count the number of tokens in the text
    # enc = tiktoken.encoding_for_model("gpt-4o")
    # tokens = len(enc.encode(text.strip()))
    # print(text)
    # print(tokens)

    # # Append the token count to the list
    # average_tokens.append(tokens)

# Calculate the average token count
average_token_count = sum(average_tokens) / len(average_tokens)
print(f"Average token count: {average_token_count}")
