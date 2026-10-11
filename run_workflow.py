"""
Compositional harness: each BIG task runs the 3-step pipeline. For every step the agent
recalls one sub-skill from memory and applies it; the step is graded against ground truth,
and the memory learns whether that recall helped. The big task succeeds only if all 3 steps
succeed — so RAG's noisy recall compounds into near-total failure, while Mnemo learns which
sub-skills to trust and the big task starts completing.
"""
import argparse
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from openai import OpenAI                              # noqa: E402
from workflow_tasks import STEPS, step_doc, make_records, make_skill_bank  # noqa: E402
from memory import embed, LocalStore, AtlasStore, Memory  # noqa: E402

_c = OpenAI()


def apply_method(method, values):
    vals = ", ".join(f"{k}={v}" for k, v in values.items())
    r = _c.chat.completions.create(
        model="gpt-4o-mini", max_tokens=12, temperature=0,
        messages=[{"role": "system", "content": "Apply the formula to the values. Reply with ONLY the number, 2 decimals."},
                  {"role": "user", "content": f"Formula: {method}\nValues: {vals}\nResult:"}])
    try:
        return round(float(r.choices[0].message.content.strip().replace("$", "")), 2)
    except Exception:
        return None


def run(use_usefulness, rounds, store_factory, label, bank, bank_vecs, recs, step_vecs):
    import time
    store = store_factory()
    store.clear()
    for b, v in zip(bank, bank_vecs):
        store.add({**b, "vec": list(v), "uses": 0, "helped": 0, "usefulness": 0.5})
    if isinstance(store, AtlasStore):
        time.sleep(10)                                      # let Atlas index the new vectors
    mem = Memory(store, use_usefulness=use_usefulness)
    accs = []
    print(f"[{label}]  usefulness={use_usefulness}")
    for r in range(rounds):
        big_ok = 0
        for rec in recs:
            known = {**rec["inputs"], **rec["truth"]}       # feed true intermediates -> clean per-step credit
            all_ok = True
            for name, ins, _m, _fn in STEPS:
                cands = mem.retrieve(step_vecs[name], k=1)
                if not cands:
                    all_ok = False; continue
                skill = cands[0]
                pred = apply_method(skill["method"], {k: known[k] for k in ins})
                ok = pred is not None and abs(pred - rec["truth"][name]) <= 0.01
                if use_usefulness:
                    mem.record([skill], ok)                 # learn: did this sub-skill help?
                all_ok = all_ok and ok
            big_ok += all_ok
        if use_usefulness:
            mem.prune()
        accs.append(big_ok / len(recs))
        print(f"  round {r+1}: big-task success {accs[-1]:5.0%}")
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--atlas", action="store_true")
    args = ap.parse_args()

    recs = make_records()
    bank = make_skill_bank()
    print(f"{len(recs)} big tasks | {len(bank)} sub-skill memories "
          f"({sum(b['truth'] for b in bank)} correct / {sum(not b['truth'] for b in bank)} noise)\n")

    bank_vecs = embed([b["document"] for b in bank])
    step_vecs = {name: v for (name, ins, _m, _fn), v in
                 zip(STEPS, embed([step_doc(n, i) for n, i, _m, _fn in STEPS]))}

    store_factory = LocalStore
    if args.atlas:
        uri = os.environ["MONGODB_URI"]
        store_factory = lambda: AtlasStore(uri)            # noqa: E731

    mnemo = run(True, args.rounds, store_factory, "Mnemo", bank, bank_vecs, recs, step_vecs)
    print()
    rag = run(False, args.rounds, store_factory, "RAG (similarity only)", bank, bank_vecs, recs, step_vecs)

    print("\n round |  Mnemo | RAG")
    for i, (m, g) in enumerate(zip(mnemo, rag), 1):
        print(f"   {i:2d}  |  {m:4.0%} | {g:4.0%}")
    print(f"\n final: Mnemo {mnemo[-1]:.0%}  vs  RAG {rag[-1]:.0%}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = range(1, len(mnemo) + 1)
        plt.figure(figsize=(7, 4.2))
        plt.plot(xs, [m * 100 for m in mnemo], "o-", lw=2.5, color="#2a7", label="Mnemo (learns which sub-skills to trust)")
        plt.plot(xs, [g * 100 for g in rag], "s--", lw=2, color="#999", label="RAG (similarity only)")
        plt.xlabel("round"); plt.ylabel("big-task completion (%)"); plt.ylim(0, 100)
        plt.title("Compositional tasks: trusting the right sub-skills compounds")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/workflow_curve.png", dpi=130)
        print(" wrote results/workflow_curve.png")
    except Exception as e:
        print("chart skipped:", e)


if __name__ == "__main__":
    main()
