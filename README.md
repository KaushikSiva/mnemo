# Mnemo — memory that learns what to remember

**Not RAG.** RAG retrieves by similarity against a frozen index and never changes. Mnemo scores
every memory by whether recalling it actually led to task **success**, reranks retrieval on that
signal, and **forgets** the memories that keep misleading. The same store gets measurably better
the more it's used — recursive self-improvement you can graph.

> Built at the MongoDB "Harness Engineering & Model Wrangling" Hackathon (SF Tech Week, Oct 11 2026).
> Everything in this repo was written during the event.

## The result

Same memory bank, same tasks, same agent. The only difference is whether recall **learns from
outcomes**. Mnemo climbs from 40% → 90%; a plain-similarity index (RAG) stays flat at 40%.

![curve](results/curve.png)

## How it works

```
task ─▶ recall from MongoDB Atlas ($vectorSearch) ─▶ agent solves ─▶ graded
                     ▲                                              │
                     └──────── usefulness update + forgetting ◀─────┘
          score = similarity × (1 + α · usefulness),  usefulness = helped / uses
```

- **Recall:** Atlas Vector Search returns candidates by similarity, then Mnemo **reranks** by
  `similarity × (1 + α·usefulness)`, so memories that have actually helped rise to the top.
- **Learn:** after each task, every recalled memory's `usefulness = helped / uses` is updated from
  the outcome. (This is the step RAG doesn't have.)
- **Forget:** memories that have been tried enough and keep failing are **pruned** from Atlas.

That feedback loop is the "recursive self-improvement": better recall → more success → a better
usefulness signal → better recall.

## Why this isn't "basic RAG"

RAG = embed, store, retrieve-by-similarity, answer. The index is **static**; it has no notion of
whether a retrieved chunk was any good. Mnemo adds a closed outcome loop on top of Atlas vector
search: it **learns a per-memory usefulness from downstream task success, reranks on it, and
forgets the bad entries**. Remove that loop and you get the flat grey line in the chart.

## The benchmark (how we *measure* memory quality)

A controlled task set where success requires recalling the right past experience. Each task is a
business document; the number to return follows an **arbitrary per-family rule** the model can't
guess zero-shot (INVOICE → subtotal+tax, QUOTE → subtotal−discount, …). The memory bank holds the
correct "skill" for each family **plus realistic noise** — stale/incorrect skills, which look
textually identical. Plain similarity can't tell the good memory from the bad; only outcomes can.

- `tasks.py` — benchmark: families, rules, held-out eval set, noisy memory bank.
- `memory.py` — the engine: `LocalStore` + `AtlasStore` (`$vectorSearch`), usefulness + pruning.
- `agent.py` — applies the recalled method (deliberately simple; the smarts are in recall).
- `run.py` — round harness, the Mnemo-vs-RAG ablation, the chart.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install openai pymongo python-dotenv numpy matplotlib

cp .env.example .env    # fill in OPENAI_API_KEY and the Atlas MONGODB_URI
python setup_atlas.py   # one-time: creates the vector-search index
python run.py --atlas   # runs on MongoDB Atlas (the submission path)
python run.py           # local vector store, for quick iteration
```

Built with **MongoDB Atlas Vector Search** + OpenAI (`gpt-4o-mini`, `text-embedding-3-small`).
