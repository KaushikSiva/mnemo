"""
One-time Atlas setup: creates the vector-search index on the memories collection.
Run once after MONGODB_URI is set:  python setup_atlas.py
"""
import os
import time

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DB, COLL, INDEX = "mnemo", "memories", "mnemo_vec"


def main():
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise SystemExit("set MONGODB_URI in .env first")
    col = MongoClient(uri)[DB][COLL]
    col.insert_one({"_warmup": True})          # ensure the collection exists
    col.delete_many({"_warmup": True})

    existing = [i["name"] for i in col.list_search_indexes()]
    if INDEX in existing:
        print(f"index '{INDEX}' already exists")
        return
    col.create_search_index(SearchIndexModel(
        name=INDEX, type="vectorSearch",
        definition={"fields": [{"type": "vector", "path": "vec",
                                "numDimensions": 1536, "similarity": "cosine"}]}))
    print(f"creating index '{INDEX}' … (Atlas takes ~1-2 min to build)")
    while True:
        idx = next((i for i in col.list_search_indexes() if i["name"] == INDEX), None)
        if idx and idx.get("queryable"):
            print("index ready."); break
        time.sleep(5)


if __name__ == "__main__":
    main()
