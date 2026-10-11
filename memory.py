"""
Mnemo — a self-improving memory engine.

Not RAG. RAG retrieves by similarity against a frozen index. Mnemo:
  - scores every memory by whether recalling it actually led to task SUCCESS (usefulness),
  - reranks retrieval by  similarity * (1 + alpha * usefulness),
  - forgets (prunes) memories that keep misleading.
So the same store gets measurably better the more it's used.

Two backends, one interface: LocalStore (dev, numpy cosine) and AtlasStore
(MongoDB Atlas $vectorSearch — required for the hackathon submission).
"""
import os

import numpy as np
from openai import OpenAI

_client = OpenAI()
EMBED_MODEL = "text-embedding-3-small"
DIM = 1536


def embed(texts):
    r = _client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in r.data]


# ---------------------------------------------------------------- backends

class LocalStore:
    """In-process vector store (cosine). For development only."""
    def __init__(self):
        self.docs = []

    def add(self, doc):
        self.docs.append(doc)

    def search(self, qvec, n):
        if not self.docs:
            return []
        V = np.array([d["vec"] for d in self.docs])
        q = np.array(qvec)
        sims = V @ q / (np.linalg.norm(V, axis=1) * np.linalg.norm(q) + 1e-9)
        idx = np.argsort(-sims)[:n]
        return [(self.docs[i], float(sims[i])) for i in idx]

    def all(self):
        return self.docs

    def replace_all(self, docs):
        self.docs = docs

    def clear(self):
        self.docs = []


class AtlasStore:
    """MongoDB Atlas vector store. Memories persist in a collection; recall uses $vectorSearch."""
    def __init__(self, uri, db="mnemo", coll="memories", index="mnemo_vec"):
        from pymongo import MongoClient
        self.c = MongoClient(uri)
        self.col = self.c[db][coll]
        self.index = index

    def add(self, doc):
        self.col.insert_one(doc)

    def search(self, qvec, n):
        pipeline = [
            {"$vectorSearch": {"index": self.index, "path": "vec", "queryVector": list(qvec),
                               "numCandidates": max(100, n * 10), "limit": n}},
            {"$addFields": {"sim": {"$meta": "vectorSearchScore"}}},
        ]
        return [(d, float(d.get("sim", 0.0))) for d in self.col.aggregate(pipeline)]

    def all(self):
        return list(self.col.find({}))

    def clear(self):
        self.col.delete_many({})

    def update_stats(self, _id, uses, helped, usefulness):
        self.col.update_one({"_id": _id}, {"$set": {"uses": uses, "helped": helped, "usefulness": usefulness}})

    def delete(self, _id):
        self.col.delete_one({"_id": _id})


# ---------------------------------------------------------------- the engine

class Memory:
    def __init__(self, store, alpha=4.0, use_usefulness=True):
        self.store = store
        self.alpha = alpha                 # how hard usefulness bends the ranking
        self.use_usefulness = use_usefulness   # False = ablation (plain similarity, i.e. RAG)

    def retrieve(self, qvec, k=3, pool=24):
        cands = self.store.search(qvec, pool)
        scored = []
        for doc, sim in cands:
            u = doc.get("usefulness", 0.5)
            score = sim * (1 + self.alpha * u) if self.use_usefulness else sim
            scored.append((score, doc))
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[:k]]

    def add(self, task, demo_answer, qvec, round_i, correct):
        self.store.add({
            "document": task["document"], "question": task["question"],
            "demo_answer": str(demo_answer), "family": task["family"],   # family stored for analysis only
            "vec": list(qvec), "uses": 0, "helped": 0, "usefulness": 0.5,
            "round": round_i, "was_correct": bool(correct),
        })

    def record(self, used_docs, success):
        """The core learning step: did recalling these memories lead to success?"""
        for d in used_docs:
            d["uses"] = d.get("uses", 0) + 1
            if success:
                d["helped"] = d.get("helped", 0) + 1
            d["usefulness"] = (d["helped"] + 1) / (d["uses"] + 2)   # smoothed success rate
            if isinstance(self.store, AtlasStore) and "_id" in d:
                self.store.update_stats(d["_id"], d["uses"], d["helped"], d["usefulness"])

    def prune(self, min_uses=2, min_usefulness=0.35):
        """Forget memories that have been tried enough and keep misleading."""
        removed = 0
        if isinstance(self.store, AtlasStore):
            for d in self.store.all():
                if d.get("uses", 0) >= min_uses and d.get("usefulness", 0.5) < min_usefulness:
                    self.store.delete(d["_id"]); removed += 1
        else:
            keep = [d for d in self.store.all()
                    if not (d.get("uses", 0) >= min_uses and d.get("usefulness", 0.5) < min_usefulness)]
            removed = len(self.store.all()) - len(keep)
            self.store.replace_all(keep)
        return removed
