import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FileLoader:
    """Класс для загрузки и обработки файлов разных форматов"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        from config import Config

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or Config.CHUNK_SIZE,
            chunk_overlap=chunk_overlap or Config.CHUNK_OVERLAP,
            length_function=len,
        )

    def load_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Загрузка одного файла"""
        file_extension = Path(file_path).suffix.lower()

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension in [".html", ".htm"]:
                loader = UnstructuredHTMLLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {file_extension}")

            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)

            for chunk in chunks:
                chunk.metadata["source_file"] = os.path.basename(file_path)
                chunk.metadata["file_path"] = file_path

            return chunks

        except Exception as e:
            print(f"Ошибка при загрузке файла {file_path}: {e}")
            return []

    def load_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Загрузка всех файлов из директории"""
        all_chunks = []
        supported_extensions = [".pdf", ".docx", ".doc", ".html", ".htm", ".txt"]

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(root, file)
                    chunks = self.load_file(file_path)
                    all_chunks.extend(chunks)

        return all_chunks
