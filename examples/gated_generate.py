"""Gate first, generate only if the gate says ANSWER. Both calls hit the one engine."""
import json

from evidence_gate import gate
from generate import chat


def answer(question, passages, retrieval_available=True):
    g = gate(question, passages, retrieval_available)
    if g["action"] != "ANSWER":
        return {"action": g["action"], "margin": g["margin"], "answer": None}
    ctx = "\n".join(f"[{i + 1}] {p['title']}\n{p['text']}" for i, p in enumerate(passages))
    text = chat(f"Question: {question}\n\nRetrieved evidence:\n{ctx}\n\nAnswer in one or two sentences using only the evidence.")
    return {"action": "ANSWER", "margin": g["margin"], "answer": text}


if __name__ == "__main__":
    print(json.dumps(answer("Who wrote the novel?", [{"title": "Note", "text": "The novel was written by A. Writer."}]), indent=1))
    print(json.dumps(answer("When was the bridge built?", [{"title": "Bridge", "text": "The bridge is long."}], False), indent=1))
