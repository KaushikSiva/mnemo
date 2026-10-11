"""
A controlled benchmark for measuring MEMORY quality.

Each task is a short business doc + a question. The number you must return is set by an
ARBITRARY per-family rule the model can't guess zero-shot (INVOICE -> subtotal+tax, etc.).
The ONLY way to answer is to recall a worked example that demonstrates the SAME family's rule.

We evaluate on fresh held-out tasks, against a memory bank that contains both CORRECT
examples and realistic NOISE (entries that demonstrate the wrong rule — the stale/incorrect
memories every long-lived agent accumulates). A good memory recalls the correct ones.

Deterministic, auto-gradable (numeric exact match).
"""
import random

# distinct templates per family, so same-family docs cluster in embedding space
FAMILIES = {
    "INVOICE":   ("INVOICE",         "Bill To",        "subtotal + tax",            lambda f: f["subtotal"] + f["tax"]),
    "QUOTE":     ("PRICE QUOTE",     "Prepared For",   "subtotal - discount",       lambda f: f["subtotal"] - f["discount"]),
    "RECEIPT":   ("STORE RECEIPT",   "Served",         "largest single line item",  lambda f: max(f["items"])),
    "STATEMENT": ("ACCOUNT STATEMENT","Account Holder","sum of the last two items", lambda f: f["items"][-1] + f["items"][-2]),
    "ORDER":     ("PURCHASE ORDER",  "Vendor",         "subtotal doubled",          lambda f: f["subtotal"] * 2),
}
FAMS = list(FAMILIES)


def _fields(rng):
    items = [round(rng.uniform(10, 200), 2) for _ in range(rng.randint(2, 4))]
    return {
        "num": rng.randint(1000, 9999), "cust": rng.choice(
            ["Acme Co", "Globex", "Initech", "Umbrella", "Soylent", "Hooli", "Stark Ind", "Wayne LLC"]),
        "items": items, "subtotal": round(sum(items), 2),
        "tax": round(sum(items) * rng.choice([0.05, 0.08, 0.1]), 2),
        "discount": round(rng.uniform(5, 30), 2),
    }


def _doc(fam, f):
    title, who, _, _ = FAMILIES[fam]
    lines = [title, f"{who}: {f['cust']}   No. {f['num']}"]
    for i, amt in enumerate(f["items"], 1):
        lines.append(f"  line {i} .......... ${amt:.2f}")
    lines += [f"Subtotal ...... ${f['subtotal']:.2f}", f"Tax ........... ${f['tax']:.2f}",
              f"Discount ...... ${f['discount']:.2f}"]
    return "\n".join(lines)


def _answer(fam, f):
    return round(FAMILIES[fam][3](f), 2)


def make_eval(per_family=6, seed=100):
    rng = random.Random(seed)
    tasks = []
    for fam in FAMS:
        for _ in range(per_family):
            f = _fields(rng)
            tasks.append({"id": f"{fam}-{rng.randint(10000,99999)}", "family": fam,
                          "document": _doc(fam, f),
                          "question": "What is the amount payable? Answer with only the number, 2 decimals.",
                          "answer": _answer(fam, f)})
    rng.shuffle(tasks)
    return tasks


# the learned "skill" each memory carries: how to compute the payable for this doc type
METHOD = {
    "INVOICE":   "payable = Subtotal + Tax",
    "QUOTE":     "payable = Subtotal - Discount",
    "RECEIPT":   "payable = the single largest line item",
    "STATEMENT": "payable = sum of the last two line items",
    "ORDER":     "payable = Subtotal * 2",
}


def make_bank(correct_per=4, poison_per=6, seed=1):
    """Memory bank: for each family, correct remembered methods + NOISE (a wrong method —
    the stale/incorrect 'skills' a long-lived agent accumulates). Same family text, so plain
    similarity can't tell the good memory from the bad one. Only outcomes can."""
    rng = random.Random(seed)
    bank = []
    for fam in FAMS:
        for _ in range(correct_per):
            bank.append({"family": fam, "document": _doc(fam, _fields(rng)),
                         "method": METHOD[fam], "truth": True})
        for _ in range(poison_per):
            wrong = rng.choice([x for x in FAMS if x != fam])
            bank.append({"family": fam, "document": _doc(fam, _fields(rng)),
                         "method": METHOD[wrong], "truth": False})
    rng.shuffle(bank)
    return bank


def grade(predicted, answer, tol=0.01):
    try:
        return abs(float(str(predicted).replace("$", "").strip()) - float(answer)) <= tol
    except Exception:
        return False


if __name__ == "__main__":
    ev, bank = make_eval(), make_bank()
    print(f"{len(ev)} eval tasks, {len(bank)} bank memories "
          f"({sum(b['truth'] for b in bank)} correct / {sum(not b['truth'] for b in bank)} noise)")
    t = ev[0]
    print("\nsample eval —", t["family"], "rule:", FAMILIES[t["family"]][2], "answer:", t["answer"])
    print(t["document"])
