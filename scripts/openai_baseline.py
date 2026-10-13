"""OpenAI baseline: same SemIf prompt, confidence = first-token logprob (top-1 only)."""

import json
import math
import sys
import time
import urllib.request

import os

sys.path.insert(0, os.path.join(os.getenv("SEMIF_REPO", "semif"), "src"))
from semif_phase1.core import LETTERS, direct_messages  # noqa: E402

MODEL, INPUT, OUTPUT = sys.argv[1], sys.argv[2], sys.argv[3]
KEY = os.environ["OPENAI_API_KEY"]


def score(row):
    body = {
        "model": MODEL,
        "messages": direct_messages(row),
        "max_completion_tokens": 4,
        "logprobs": True,
        "top_logprobs": 5,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.load(r)
    ms = (time.perf_counter() - t0) * 1000
    first = resp["choices"][0]["logprobs"]["content"][0]
    letters = LETTERS[: len(row["options"])]
    tok = first["token"].strip()
    n = len(letters)
    if tok in letters:
        # OpenAI only returns top-1: the remaining mass is split evenly across the other options.
        p = math.exp(first["logprob"])
        probs = [p if l == tok else (1 - p) / (n - 1) for l in letters]
    else:
        probs = [1 / n] * n
    return {"id": row["id"], "probs": probs, "token": tok, "ms": ms, "usage": resp["usage"]}


rows = [json.loads(l) for l in open(INPUT)]
with open(OUTPUT, "w") as out:
    for i, row in enumerate(rows):
        res = score(row)
        res["label"] = row.get("label")
        out.write(json.dumps(res) + "\n")
        out.flush()
        print(f"{i + 1}/{len(rows)} {res['ms']:.0f} ms {res['token']}", file=sys.stderr, flush=True)
