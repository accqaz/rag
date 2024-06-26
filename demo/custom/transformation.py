from typing import Sequence
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.schema import BaseNode


class CustomFilePathExtractor(BaseExtractor):
    last_path_length: int = 4

    def __init__(self, last_path_length: int = 4, **kwargs):
        super().__init__(last_path_length=last_path_length, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "CustomFilePathExtractor"

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        metadata_list = []
        for node in nodes:
            node.metadata["file_path"] = "/".join(
                node.metadata["file_path"].split("/")[-self.last_path_length :]
            )
            metadata_list.append(node.metadata)
        return metadata_list


class CustomTitleExtractor(BaseExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "CustomTitleExtractor"

    # 将Document的第一行作为标题
    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        try:
            document_title = nodes[0].text.split("\n")[0]
            last_file_path = nodes[0].metadata["file_path"]
        except:
            document_title = ""
            last_file_path = ""
        metadata_list = []
        for node in nodes:
            if node.metadata["file_path"] != last_file_path:
                document_title = node.text.split("\n")[0]
                last_file_path = node.metadata["file_path"]
            node.metadata["document_title"] = document_title

            # 提取并设置document_type
            file_path = node.metadata["file_path"]
            try:
                path_parts = file_path.split("/")
                #print(f"path_parts: {path_parts}")
                data_index = path_parts.index("data")
                document_type = path_parts[data_index + 2] if data_index + 2 < len(path_parts) else "unknown"
                graph_type = path_parts[data_index + 1] if data_index + 1 < len(path_parts) else "unknown"
                #print(f"Document type extracted: {document_type}")
            except (ValueError, IndexError):
                document_type = "unknown"
                graph_type = "unknown"
                #print("Document type extraction failed.")
            node.metadata["document_type"] = document_type
            node.metadata["graph_type"] = graph_type

            metadata_list.append(node.metadata)
        print("data has been extracted!")
        return metadata_list
