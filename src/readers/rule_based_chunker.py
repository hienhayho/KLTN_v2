import re
import json
from pathlib import Path
from llama_index.core import Document
from typing import Dict, Optional, List
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.readers.base import BaseReader

prompt = """
Bạn được cung cấp một đoạn văn gồm tập hợp những bước khác nhau trong quy trình. Tuy nhiên, các bước này có thể thuộc cùng 1 quy trình hoặc nhiều quy trình. Điểm nhận biết khi có nhiều quy trình là có nhiều B1, tên quy trình sẽ trên B1 đó. Hãy giúp tôi trích xuất ra nội dung của các quy trình khác nhau trong đoạn văn này theo JSON format sau:

{{
    "titles": ["<tên quy trình 1>", "<tên quy trình 2>", ...],
    "processes: ["<nội dung đầy đủ không chỉnh sửa của quy trình 1>", "<nội dung đầy đủ không chỉnh sửa của quy trình 2>", ...]
}}
"""

llm = OpenAI(model="gpt-4o-mini", system_prompt=prompt)


class CustomReader(BaseReader):
    def __init__(self, titles: list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.titles = titles

    def clean_newlines(self, text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", text)

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs=None,
    ) -> List[Document]:
        """Parse file."""
        if not isinstance(file, Path):
            file = Path(file)

        try:
            import docx2txt
        except ImportError:
            raise ImportError(
                "docx2txt is required to read Microsoft Word files: "
                "`pip install docx2txt`"
            )

        if fs:
            with fs.open(str(file)) as f:
                text = docx2txt.process(f)
        else:
            text = docx2txt.process(file)
        metadata = {
            "file_name": file.name,
            "law_name": file.parent.name,
            "doc_name": file.name,
        }
        if extra_info is not None:
            metadata.update(extra_info)

        # Find index of each title in the document
        indexes = []
        for title in self.titles:
            index = text.find(title)
            if index != -1:
                indexes.append(index)
        indexes.append(len(text))
        indexes[-2] = indexes[-2] - 1

        documents = []
        idx = indexes[0]
        for i in range(len(indexes) - 1):
            if i == 4:
                # print(self.titles[i])
                documents.extend(
                    self._split_content(
                        text=text[idx : indexes[i + 1]],
                        law_name=file.parent.name,
                        doc_name=file.name,
                    )
                )
                idx = indexes[i + 1]
                continue

            documents.append(
                # f"Bộ luật: {file.parent.name} - Tài liệu: {file.name}"
                # + "\n"
                # + text[idx : indexes[i + 1]]
                text[idx : indexes[i + 1]]
            )
            idx = indexes[i + 1]

        return [
            Document(text=self.clean_newlines(text), metadata=metadata or {})
            for text in documents
        ]

    def _split_content(self, text: str, law_name: str, doc_name: str) -> List[str]:
        idx = 1
        indexes = []
        # print("text: ", text)
        while True:
            title_to_extract = text.find(f"5.{idx}")
            if title_to_extract == -1:
                break
            indexes.append(title_to_extract)
            idx += 1

        indexes.append(len(text))

        documents = []
        for i in range(len(indexes) - 1):
            if i == 6:
                titles, processes = self._split_large_process(
                    text=text[indexes[i] : indexes[i + 1]]
                )
                for title, process in zip(titles, processes):
                    documents.extend(
                        self._split_process(
                            text=process,
                            title=title,
                            law_name=law_name,
                            doc_name=doc_name,
                        )
                    )
                continue

            documents.append(
                # f"Bộ luật: {law_name} - Tài liệu: {doc_name} - "
                # + "\n"
                # + text[indexes[i] : indexes[i + 1]]
                text[indexes[i] : indexes[i + 1]]
            )

        return documents

    def _split_process(
        self, text: str, title: str, law_name: str, doc_name: str
    ) -> List[str]:
        # idx = 1
        # indexes = []
        # while idx < 10:
        #     title_to_extract = text.find(f"B{idx}")
        #     indexes.append(title_to_extract)
        #     idx += 1

        # indexes.append(len(text))

        # documents = []
        # for i in range(len(indexes) - 1):
        #     documents.append(
        #         f"Bộ luật: {law_name} - Tài liệu: {doc_name} - Quy trình xử lý: {title}"
        #         + "\n"
        #         + text[indexes[i] : indexes[i + 1]]
        #     )
        documents = [
            f"Bộ luật: {law_name} - Tài liệu: {doc_name} - Quy trình xử lý: {title}"
            + "\n"
            + text
            # text
        ]

        return documents

    def _split_large_process(self, text: str):
        messages = [
            ChatMessage(
                role="system",
                content=prompt,
            ),
            ChatMessage(role="user", content=text),
        ]

        response = llm.chat(messages=messages, response_format={"type": "json_object"})
        result = json.loads(response.message.content)
        return result["titles"], result["processes"]
