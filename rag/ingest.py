import json

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

_EMBEDDINGS_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")

_CHROMA_PATH = "rag/chroma_db"
_COLLECTION_NAME = "waste_disposal_items"

_ITEMS_PATH = "rag/data/waste_items.json"


def _build_documents(items):
    documents = []
    ids = []

    for item in items:
        for i, classification in enumerate(item["분류"]):
            metadata = {
                "item_name": item["품목명"],
                "disposal_method": item["배출방법"],
                "features": item["특징"],
            }
            for json_key, meta_key in (("카테고리", "category"), ("소분류", "subcategory"), ("세부", "detail")):
                value = classification.get(json_key)
                if value:
                    metadata[meta_key] = value

            documents.append(Document(page_content=item["품목명"], metadata=metadata))
            ids.append(f"{item['id']}-{i}")

            for j, synonym in enumerate(item.get("유의어", [])):
                documents.append(Document(page_content=synonym, metadata=metadata))
                ids.append(f"{item['id']}-{i}-syn-{j}")

    return documents, ids


def ingest():
    with open(_ITEMS_PATH, encoding="utf-8") as f:
        items = json.load(f)

    documents, ids = _build_documents(items)

    vectorstore = Chroma(
        collection_name=_COLLECTION_NAME,
        embedding_function=_EMBEDDINGS_MODEL,
        persist_directory=_CHROMA_PATH,
        collection_metadata={"hnsw:space": "cosine"},
    )

    vectorstore.add_documents(documents, ids=ids)

if __name__ == "__main__":
    ingest()
