"""
Compositional use case: a WORKFLOW agent completes a BIG task by recalling the small
SUB-SKILLS it needs. The big task = a 3-step pipeline (subtotal -> net -> total). For each
step the agent recalls a skill from memory and applies it. The big task succeeds only if
EVERY recalled sub-skill is the right one — so errors compound, and knowing which small
memories to trust is what makes the big task complete.

Memory holds, per sub-skill, the correct method plus realistic NOISE (a plausible-but-wrong
method). Within a skill the entries look identical, so similarity can't tell good from bad —
only outcomes can. Deterministic + auto-gradable.
"""
import random

# the pipeline: each step reads named values and produces the next one
STEPS = [
    ("subtotal", ("base", "qty"),       "subtotal = base * qty",          lambda v: v["base"] * v["qty"]),
    ("net",      ("subtotal", "rate"),  "net = subtotal * (1 - rate)",    lambda v: v["subtotal"] * (1 - v["rate"])),
    ("total",    ("net", "fee"),        "total = net + fee",              lambda v: v["net"] + v["fee"]),
]
WRONG = {  # plausible-but-wrong sub-skill (noise)
    "subtotal": "subtotal = base + qty",
    "net":      "net = subtotal - rate",
    "total":    "total = net * fee",
}


STEP_DESC = {   # distinctive wording per sub-skill so recall targets the right one
    "subtotal": "SUBTOTAL skill: multiply the base price by the quantity ordered.",
    "net":      "NET skill: apply the discount rate to the subtotal amount.",
    "total":    "TOTAL skill: add the delivery fee to the net amount.",
}


def step_doc(step, inputs):
    return STEP_DESC[step]


def make_records(n=12, seed=100):
    rng = random.Random(seed)
    recs = []
    for _ in range(n):
        v = {"base": round(rng.uniform(5, 50), 2), "qty": rng.randint(2, 9),
             "rate": rng.choice([0.05, 0.1, 0.15, 0.2]), "fee": round(rng.uniform(2, 20), 2)}
        truth = {}
        vv = dict(v)
        for name, _ins, _m, fn in STEPS:
            vv[name] = round(fn(vv), 2)
            truth[name] = vv[name]
        recs.append({"inputs": v, "truth": truth, "answer": truth["total"]})
    return recs


def _example(ins, rng):
    return ", ".join(f"{k}={round(rng.uniform(3, 40), 1)}" for k in ins)


def make_skill_bank(correct_per=3, poison_per=5, seed=1):
    rng = random.Random(seed)
    bank = []
    for name, ins, method, _fn in STEPS:
        base = step_doc(name, ins)
        for _ in range(correct_per):
            # a worked example varies the embedding per entry (so the index can rank them) while
            # still clustering by sub-skill; the method is what the agent actually applies.
            doc = f"{base} e.g. with {_example(ins, rng)}, using {method}."
            bank.append({"step": name, "document": doc, "method": method, "truth": True})
        for _ in range(poison_per):
            doc = f"{base} e.g. with {_example(ins, rng)}, using {WRONG[name]}."
            bank.append({"step": name, "document": doc, "method": WRONG[name], "truth": False})
    rng.shuffle(bank)
    return bank


if __name__ == "__main__":
    recs, bank = make_records(), make_skill_bank()
    print(f"{len(recs)} big tasks, {len(bank)} sub-skill memories "
          f"({sum(b['truth'] for b in bank)} correct / {sum(not b['truth'] for b in bank)} noise)")
    r = recs[0]
    print("\nbig task inputs:", r["inputs"])
    print("pipeline truth:", r["truth"], "-> answer", r["answer"])
