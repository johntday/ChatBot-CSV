import os
import shutil

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from typing import Any, List
from langchain.docstore.document import Document
from modules.MyNotionDBLoader import MyNotionDBLoader
from pathlib import Path
import argparse

DB_NAME = 'notion_hybris_faiss_index'
DB_NAME2 = 'notion_hybris_faiss_index2'


def dir_path_str(name=DB_NAME) -> str:
    return f"embeddings/{name}/"


def load_notion_documents(notion_token, notion_database_id) -> List[Document]:
    loader = MyNotionDBLoader(notion_token, notion_database_id)
    documents = loader.load()
    print(f"\nLoaded {len(documents)} documents from Notion")
    # if len(documents) > 0:
    #     print(f"\nFirst document: {documents[0]}")
    return documents


def load_pdf_documents() -> List[Document]:
    loader = DirectoryLoader('docs', glob="**/*.pdf", loader_cls=PyPDFLoader)
    pages = loader.load_and_split()
    return pages


def replace_non_ascii(doc: Document) -> Document:
    """
    Replaces non-ascii characters with ascii characters
    """
    return Document(page_content="".join([i if ord(i) < 128 else " " for i in doc.page_content]), metadata=doc.metadata)


def split_documents(documents) -> List[Document]:
    clean_documents = [replace_non_ascii(doc) for doc in documents]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
        length_function=len,
    )
    document_chunks = text_splitter.split_documents(clean_documents)
    return document_chunks


def faiss_load_vector_store(documents, name=DB_NAME) -> FAISS:
    if len(documents) == 0:
        raise ValueError("\nNo documents to save")

    s_dir_path = dir_path_str(name)
    print(f"\nSaving '{s_dir_path}'")

    vector_db = FAISS.from_documents(documents, OpenAIEmbeddings())

    dir_path = Path(s_dir_path)
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
        print(f"\nRemoved '{s_dir_path}'")

    vector_db.save_local(s_dir_path)
    print(f"\nSaved '{s_dir_path}'")
    return vector_db


def faiss_update_vector_store_deprecated(documents):
    if len(documents) == 0:
        raise ValueError("\nNo documents to save")

    db2 = FAISS.from_documents(documents, OpenAIEmbeddings())
    faiss_merge_vector_stores(None, db2)


def faiss_update_vector_store(documents):
    db = faiss_fetch_vector_store()

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    db.add_texts(texts, metadatas)

    s_dir_path = dir_path_str(DB_NAME)
    dir_path = Path(s_dir_path)
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)

    db.save_local(s_dir_path)


def faiss_merge_vector_stores(db1, db2):
    if db1 is None:
        db1 = faiss_fetch_vector_store(DB_NAME)
    if db2 is None:
        db2 = faiss_fetch_vector_store(DB_NAME2)
    db1.merge_from(db2)

    dir_path1 = Path(dir_path_str(DB_NAME))
    if dir_path1.exists() and dir_path1.is_dir():
        shutil.rmtree(dir_path1)
    dir_path2 = Path(dir_path_str(DB_NAME2))
    if dir_path2.exists() and dir_path2.is_dir():
        shutil.rmtree(dir_path2)

    db1.save_local(dir_path_str(DB_NAME))
    print("\nMerged finished")


def faiss_fetch_vector_store(name=DB_NAME) -> FAISS:
    vector_db = FAISS.load_local(f"embeddings/{name}", OpenAIEmbeddings())
    print(f"\nLoaded '{name}'")
    return vector_db


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("mode", choices=['notion', 'docs'], help="mode is either 'notion' or 'docs'")
    # args = parser.parse_args()
    # print(args.echo)

    load_dotenv(verbose=True)
    # merge_vector_stores()

    # if args.mode == 'notion':
    NOTION_TOKEN = os.getenv("NOTION_TOKEN")
    NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
    docs = load_notion_documents(notion_token=NOTION_TOKEN, notion_database_id=NOTION_DATABASE_ID)
    doc_chunks = split_documents(docs)

    faiss_load_vector_store(doc_chunks)

    # faiss_update_vector_store(doc_chunks)

    # elif args.mode == 'docs':
    #     docs = load_pdf_documents()
    #     doc_chunks = split_documents(docs)
    #     name = 'hybris_pdf_faiss_index'
    #     db = load_vector_store(doc_chunks, name)
    # elif print('error: invalid mode'):
    #     exit(1)
