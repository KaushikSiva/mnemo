"""The agent: apply the recalled method to the document. Deliberately simple — the
intelligence is in WHICH memory gets recalled, which is the memory engine's job."""
from openai import OpenAI

_c = OpenAI()
SYS = ("You compute one number from a business document using the METHOD you are given. "
       "Fields in the document: Subtotal, Tax, Discount, and the line items. "
       "Apply the method exactly. Reply with ONLY the number, 2 decimals, nothing else.")


def solve(task, examples):
    if examples:
        method = examples[0].get("method", "")
        user = f"METHOD: {method}\n\nDOCUMENT:\n{task['document']}\n\nCompute the payable."
    else:
        user = f"DOCUMENT:\n{task['document']}\n\nCompute the payable."
    r = _c.chat.completions.create(
        model="gpt-4o-mini", max_tokens=12, temperature=0,
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}])
    return r.choices[0].message.content.strip()
