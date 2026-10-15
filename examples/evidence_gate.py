"""Evidence gate through the same engine (gate front, :8100): ANSWER / RETRIEVE_MORE / ABSTAIN."""
import json
import urllib.request

GATE = "http://127.0.0.1:8100"


def gate(question, passages, retrieval_available=True):
    body = {"question": question, "passages": passages, "retrieval_available": retrieval_available}
    req = urllib.request.Request(f"{GATE}/v1/gate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


if __name__ == "__main__":
    out = gate("Who wrote the novel?", [{"title": "Note", "text": "The novel was written by A. Writer."}])
    print(json.dumps({k: out[k] for k in ("action", "margin", "tau", "probs")}, indent=1))
