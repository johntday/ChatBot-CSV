import os
import shutil

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from typing import Any, List
from langchain.docstore.document import Document
from modules.MyNotionDBLoader import MyNotionDBLoader
from pathlib import Path
import argparse

DB_NAME = 'notion_hybris_faiss_index'
DB_NAME2 = 'notion_hybris_faiss_index2'


def load_notion_documents(notion_token, notion_database_id) -> List[Document]:
    loader = MyNotionDBLoader(notion_token, notion_database_id)
    documents = loader.load()
    print(f"\nLoaded {len(documents)} documents from Notion")
    print(f"\nFirst document: {documents[0]}")
    return documents


def load_pdf_documents() -> List[Document]:
    loader = DirectoryLoader('docs', glob="**/*.pdf", loader_cls=PyPDFLoader)
    pages = loader.load_and_split()
    return pages


def split_documents(documents) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
        length_function=len,
    )
    document_chunks = text_splitter.split_documents(documents)
    return document_chunks


def load_vector_store(documents, name) -> FAISS:
    s_dir_path = f"embeddings/{name}/"
    print(f"\nSaving '{s_dir_path}'")

    vector_db = FAISS.from_documents(documents, OpenAIEmbeddings())

    dir_path = Path(s_dir_path)
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
        print(f"\nRemoved '{s_dir_path}'")

    vector_db.save_local(s_dir_path)
    print(f"\nSaved '{s_dir_path}'")
    return vector_db


def merge_vector_stores():
    db1 = fetch_vector_store("notion_hybris_faiss_index")
    db2 = fetch_vector_store("notion_hybris_faiss_index2")
    db1.merge_from(db2)

    dir_path = Path("embeddings/notion_hybris_faiss_index/")
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
    dir_path = Path("embeddings/notion_hybris_faiss_index2/")
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)

    db1.save_local("embeddings/notion_hybris_faiss_index")
    print("\nMerged finished")


def fetch_vector_store(name=DB_NAME) -> FAISS:
    vector_db = FAISS.load_local(f"embeddings/{name}", OpenAIEmbeddings())
    print(f"\nLoaded '{name}'")
    return vector_db


def test_search():
    db = fetch_vector_store(DB_NAME)
    query = "list of hybris capabilities"
    results = db.similarity_search(query, k=1)
    print(results)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("mode", choices=['notion', 'docs'], help="mode is either 'notion' or 'docs'")
    # args = parser.parse_args()
    # print(args.echo)

    load_dotenv(verbose=True)

    merge_vector_stores()
    exit(0)

    # if args.mode == 'notion':
    NOTION_TOKEN = os.getenv("NOTION_TOKEN")
    # NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

    NOTION_DATABASE_ID = 'db0ee43b057247c9a897d8dd57ff34a3'
    docs = load_notion_documents(notion_token=NOTION_TOKEN, notion_database_id=NOTION_DATABASE_ID)
    doc_chunks = split_documents(docs)
    """ CHECK ON DB_NAME2 """
    db = load_vector_store(doc_chunks, DB_NAME2)

    # elif args.mode == 'docs':
    #     docs = load_pdf_documents()
    #     doc_chunks = split_documents(docs)
    #     name = 'hybris_pdf_faiss_index'
    #     db = load_vector_store(doc_chunks, name)
    # elif print('error: invalid mode'):
    #     exit(1)


    # merge_vector_stores()
