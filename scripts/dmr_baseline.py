"""Baseline: mismo prompt de SemIf, leído vía logprobs del primer token en DMR (OpenAI-compatible)."""

import json
import math
import sys
import time
import urllib.request

import os

sys.path.insert(0, os.path.join(os.getenv("SEMIF_REPO", "semif"), "src"))
from semif_phase1.core import LETTERS, direct_messages  # noqa: E402

URL = "http://localhost:12434/engines/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "ai/qwen3.5:9b-q4_K_XL"
INPUT, OUTPUT = sys.argv[2], sys.argv[3]


def score(row):
    body = {
        "model": MODEL,
        "messages": direct_messages(row),
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    ms = (time.perf_counter() - t0) * 1000
    top = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    letters = LETTERS[: len(row["options"])]
    # Logprob por letra; si no aparece en el top-20 queda en -inf (prob 0).
    lp = {l: -math.inf for l in letters}
    for t in top:
        tok = t["token"].strip()
        if tok in lp:
            lp[tok] = max(lp[tok], t["logprob"])
    mass = [math.exp(lp[l]) for l in letters]
    total = sum(mass)
    probs = [m / total for m in mass] if total > 0 else [1 / len(letters)] * len(letters)
    return {"id": row["id"], "probs": probs, "covered_mass": total, "ms": ms}


import re
import subprocess

SWAP_GROWTH_MB, SERVER_MAX_GB, MIN_FREE_PCT = 3000, 8.5, 15


def swap_used_mb():
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(re.search(r"used = ([\d.]+)M", out).group(1))


def server_rss_gb():
    out = subprocess.run(["ps", "-axo", "rss,comm"], capture_output=True, text=True).stdout
    return sum(int(l.split()[0]) for l in out.splitlines() if "llama-server" in l) / 1048576


def free_pct():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    return int(re.search(r"free percentage: (\d+)%", out).group(1))


def guard(swap0):
    # Sin reintentos: cualquier señal de presión de memoria aborta.
    swap, rss, free = swap_used_mb(), server_rss_gb(), free_pct()
    if swap - swap0 > SWAP_GROWTH_MB or rss > SERVER_MAX_GB or free < MIN_FREE_PCT:
        raise SystemExit(f"ABORT memoria: swap +{swap - swap0:.0f} MB, llama-server {rss:.1f} GB, libre {free}%")
    return swap, rss, free


rows = [json.loads(l) for l in open(INPUT)]
swap0 = swap_used_mb()
# Reanuda: salta ids ya escritos.
try:
    done = {json.loads(l)["id"] for l in open(OUTPUT)}
except FileNotFoundError:
    done = set()
with open(OUTPUT, "a") as out:
    for i, row in enumerate(rows):
        if row["id"] in done:
            continue
        guard(swap0)
        res = score(row)
        res["label"] = row.get("label")
        out.write(json.dumps(res) + "\n")
        out.flush()
        swap, rss, free = guard(swap0)
        print(f"{i + 1}/{len(rows)} {res['ms']:.0f} ms | server {rss:.1f} GB, swap +{swap - swap0:.0f} MB, libre {free}%", file=sys.stderr, flush=True)
