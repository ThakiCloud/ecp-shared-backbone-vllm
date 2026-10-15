"""Plain generation through the shared engine (vLLM OpenAI API, :8000). Standard library only."""
import json
import sys
import urllib.request

ENGINE = "http://127.0.0.1:8000"


def chat(prompt, max_tokens=256):
    body = {"model": "ecp-sb", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{ENGINE}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(chat(" ".join(sys.argv[1:]) or "Explain in two sentences why the sky looks blue."))
