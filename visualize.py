"""
Visualize the recursive self-improvement: run Mnemo, snapshot every memory's learned
usefulness each round, and draw three pictures:
  1) results/viz_curve.png   — accuracy climbing (Mnemo) vs flat (RAG)
  2) results/viz_memory.png  — the memory self-organizing: correct entries light up, noise fades
  3) results/viz_loop.png    — the RSI flywheel
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

from tasks import make_eval, make_bank, grade
from memory import embed, LocalStore, Memory
from agent import solve

ROUNDS = 8
GREEN, GREY, INK, PAPER = "#1fa971", "#9aa0a6", "#14181f", "#ffffff"


def run(use_usefulness, snapshots=None):
    eval_tasks, bank = make_eval(), make_bank()
    evs = embed([t["document"] for t in eval_tasks])
    bvs = embed([b["document"] for b in bank])
    store = LocalStore()
    for b, v in zip(bank, bvs):
        store.add({**b, "vec": v, "uses": 0, "helped": 0, "usefulness": 0.5})
    mem = Memory(store, use_usefulness=use_usefulness)
    accs = []
    for r in range(ROUNDS):
        retr = [mem.retrieve(v, k=1) for v in evs]
        with ThreadPoolExecutor(max_workers=12) as ex:
            preds = list(ex.map(lambda te: solve(te[0], te[1]), zip(eval_tasks, retr)))
        c = 0
        for t, ex_docs, p in zip(eval_tasks, retr, preds):
            ok = grade(p, t["answer"])
            if use_usefulness:
                mem.record(ex_docs, ok)
            c += ok
        accs.append(c / len(eval_tasks))
        if snapshots is not None:                       # record each memory's usefulness this round
            snapshots.append([mem._useful(d) for d in store.all()])
    return accs, bank


def main():
    os.makedirs("results", exist_ok=True)
    snaps = []
    mnemo, bank = run(True, snaps)
    rag, _ = run(False)
    truths = np.array([b["truth"] for b in bank])
    U = np.array(snaps).T                               # memories x rounds

    # ---- 1) accuracy curve
    xs = range(1, ROUNDS + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, [m * 100 for m in mnemo], "o-", lw=2.6, color=GREEN, label="Mnemo (learns what to recall)")
    plt.plot(xs, [g * 100 for g in rag], "s--", lw=2, color=GREY, label="RAG (frozen index)")
    plt.ylim(0, 100); plt.xlabel("round"); plt.ylabel("task accuracy (%)")
    plt.title("Recursive self-improvement, measured"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("results/viz_curve.png", dpi=140); plt.close()

    # ---- 2) memory self-organizing (correct band on top, noise below)
    order = np.argsort(~truths)                          # correct (True) first
    Us = U[order]; split = int(truths.sum())
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    im = ax.imshow(Us, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   extent=[0.5, ROUNDS + 0.5, len(bank), 0])
    ax.axhline(split, color=INK, lw=2)
    ax.text(ROUNDS + 0.7, split / 2, "correct\nmemories", va="center", color="#176b47", fontsize=10, weight="bold")
    ax.text(ROUNDS + 0.7, split + (len(bank) - split) / 2, "noisy\nmemories", va="center", color="#9a2f2f", fontsize=10, weight="bold")
    ax.set_xlabel("round"); ax.set_ylabel("memory"); ax.set_xticks(range(1, ROUNDS + 1))
    ax.set_title("The memory learns: good recalls light up, noise fades")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.12); cb.set_label("learned usefulness")
    plt.tight_layout(); plt.savefig("results/viz_memory.png", dpi=140); plt.close()

    # ---- 3) the RSI flywheel
    fig, ax = plt.subplots(figsize=(6.2, 6.2)); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.axis("off")
    steps = [("Recall\nmemories", 90), ("Act\n(solve task)", 0), ("Learn\nwhat helped", -90), ("Forget\nthe noise", 180)]
    pts = {}
    for label, deg in steps:
        a = np.radians(deg); x, y = np.cos(a), np.sin(a); pts[label] = (x, y)
        ax.add_patch(Circle((x, y), 0.32, color=GREEN if "Learn" in label or "Forget" in label else INK, zorder=3))
        ax.text(x, y, label, ha="center", va="center", color=PAPER, fontsize=10.5, weight="bold", zorder=4)
    seq = ["Recall\nmemories", "Act\n(solve task)", "Learn\nwhat helped", "Forget\nthe noise", "Recall\nmemories"]
    for a, b in zip(seq, seq[1:]):
        x0, y0 = pts[a]; x1, y1 = pts[b]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle="arc3,rad=0.28",
                     arrowstyle="-|>", mutation_scale=22, lw=2.4, color=GREY, zorder=2,
                     shrinkA=24, shrinkB=24))
    ax.text(0, 0, "better\nrecall\nevery loop", ha="center", va="center", fontsize=12, weight="bold", color="#176b47")
    ax.set_title("Mnemo's self-improvement loop", fontsize=13, weight="bold")
    plt.tight_layout(); plt.savefig("results/viz_loop.png", dpi=140); plt.close()

    print(f"Mnemo {mnemo[-1]:.0%} vs RAG {rag[-1]:.0%}")
    print("wrote results/viz_curve.png, results/viz_memory.png, results/viz_loop.png")


if __name__ == "__main__":
    main()
