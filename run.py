"""
Round harness + the money chart.

A memory bank is seeded with correct worked examples AND realistic noise (entries that
demonstrate the wrong rule). We evaluate on fresh held-out tasks each round. After each task
we teach the memory which recalled entries actually helped. Mnemo reranks+forgets by that
signal, so its accuracy climbs; RAG (plain similarity, frozen) can't and stays flat.

Same bank, same tasks, same agent — the only variable is whether recall learns from outcomes.
"""
import argparse
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from tasks import make_eval, make_bank, grade     # noqa: E402
from memory import embed, LocalStore, AtlasStore, Memory  # noqa: E402
from agent import solve                           # noqa: E402


def run(use_usefulness, rounds, k, store_factory, label, bank, bank_vecs, eval_tasks, eval_vecs):
    store = store_factory()
    for b, v in zip(bank, bank_vecs):
        store.add({**b, "vec": list(v), "uses": 0, "helped": 0, "usefulness": 0.5})
    from concurrent.futures import ThreadPoolExecutor
    mem = Memory(store, use_usefulness=use_usefulness)
    accs = []
    print(f"[{label}]  usefulness={use_usefulness}")
    for r in range(rounds):
        retrieved = [mem.retrieve(qv, k=k) for qv in eval_vecs]        # recall with this round's memory
        with ThreadPoolExecutor(max_workers=12) as ex:                # solve in parallel
            preds = list(ex.map(lambda te: solve(te[0], te[1]), zip(eval_tasks, retrieved)))
        correct = 0
        for t, ex_docs, pred in zip(eval_tasks, retrieved, preds):
            ok = grade(pred, t["answer"])
            if use_usefulness:
                mem.record(ex_docs, ok)                              # teach memory what helped (Mnemo only)
            correct += ok
        pruned = mem.prune() if use_usefulness else 0                # RAG baseline is a frozen index
        accs.append(correct / len(eval_tasks))
        print(f"  round {r+1}: acc {accs[-1]:5.0%}   bank={len(store.all()):3d}  pruned={pruned}")
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--atlas", action="store_true")
    args = ap.parse_args()

    eval_tasks = make_eval()
    bank = make_bank()
    print(f"eval={len(eval_tasks)} tasks | bank={len(bank)} "
          f"({sum(b['truth'] for b in bank)} correct / {sum(not b['truth'] for b in bank)} noise)\n")

    eval_vecs = embed([t["document"] for t in eval_tasks])
    bank_vecs = embed([b["document"] for b in bank])

    store_factory = LocalStore
    if args.atlas:
        uri = os.environ["MONGODB_URI"]
        store_factory = lambda: AtlasStore(uri)   # noqa: E731

    mnemo = run(True, args.rounds, args.k, store_factory, "Mnemo", bank, bank_vecs, eval_tasks, eval_vecs)
    print()
    rag = run(False, args.rounds, args.k, store_factory, "RAG (similarity only)", bank, bank_vecs, eval_tasks, eval_vecs)

    print("\n round |  Mnemo | RAG")
    for i, (m, g) in enumerate(zip(mnemo, rag), 1):
        print(f"   {i:2d}  |  {m:4.0%} | {g:4.0%}")
    print(f"\n final: Mnemo {mnemo[-1]:.0%}  vs  RAG {rag[-1]:.0%}   (+{(mnemo[-1]-rag[-1])*100:.0f} pts)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = range(1, len(mnemo) + 1)
        plt.figure(figsize=(7, 4.2))
        plt.plot(xs, [m * 100 for m in mnemo], "o-", lw=2.5, color="#2a7", label="Mnemo (learns what to recall)")
        plt.plot(xs, [g * 100 for g in rag], "s--", lw=2, color="#999", label="RAG (similarity only)")
        plt.xlabel("round"); plt.ylabel("task accuracy (%)"); plt.ylim(0, 100)
        plt.title("Memory that learns beats a frozen index")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/curve.png", dpi=130)
        print(" wrote results/curve.png")
    except Exception as e:
        print("chart skipped:", e)


if __name__ == "__main__":
    main()
