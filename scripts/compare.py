"""Compara SemIf vs baseline DMR sobre authored144: accuracy, balanced acc, NLL, Brier, ECE, latencia."""

import json
import os
import math
import statistics

GOLD = {r["id"]: r for r in map(json.loads, open(os.path.join(os.getenv("SEMIF_REPO", "semif"), "benchmarks/data/authored144.jsonl")))}


def load(path, probs_key, ms):
    out = {}
    for r in map(json.loads, open(path)):
        out[r["id"]] = (r[probs_key], ms(r))
    return out


def metrics(preds):
    ids = [i for i in GOLD if i in preds]
    correct, nll, brier, conf, lat = [], [], [], [], []
    per_class = {}
    for i in ids:
        p, ms = preds[i]
        y = GOLD[i]["label"]
        k = max(range(len(p)), key=p.__getitem__)
        correct.append(k == y)
        nll.append(-math.log(max(p[y], 1e-12)))
        brier.append(sum((pj - (j == y)) ** 2 for j, pj in enumerate(p)))
        conf.append(p[k])
        lat.append(ms)
        per_class.setdefault(y, []).append(k == y)
    # ECE con 10 bins sobre la confianza del argmax.
    ece = 0.0
    for b in range(10):
        idx = [j for j, c in enumerate(conf) if b / 10 < c <= (b + 1) / 10 or (b == 0 and c == 0)]
        if idx:
            ece += len(idx) / len(conf) * abs(
                statistics.mean(correct[j] for j in idx) - statistics.mean(conf[j] for j in idx)
            )
    return {
        "n": len(ids),
        "accuracy": statistics.mean(correct),
        "balanced_acc": statistics.mean(statistics.mean(v) for v in per_class.values()),
        "nll": statistics.mean(nll),
        "brier": statistics.mean(brier),
        "ece": ece,
        "p50_ms": statistics.median(lat),
        "families": {
            f: statistics.mean(c for i, c in zip(ids, correct) if GOLD[i]["family"] == f)
            for f in sorted({GOLD[i]["family"] for i in ids})
        },
    }


systems = {
    "SemIf Qwen3.5-4B MLX 8-bit": load(
        "results/raw/semif-mlx8-authored144.jsonl", "probabilities", lambda r: r["forward_seconds"] * 1000
    ),
    "DMR Qwen3.5-4B IQ4 logprobs": load("results/raw/dmr4b-authored144.jsonl", "probs", lambda r: r["ms"]),
    "DMR Qwen3.5-9B Q4 logprobs": load("results/raw/dmr9b-authored144.jsonl", "probs", lambda r: r["ms"]),
    "OpenAI gpt-5.4-nano logprob top-1": load("results/raw/nano-authored144.jsonl", "probs", lambda r: r["ms"]),
}
for name, preds in systems.items():
    m = metrics(preds)
    fam = " ".join(f"{k[:9]}={v:.3f}" for k, v in m.pop("families").items())
    print(name, json.dumps({k: round(v, 4) for k, v in m.items()}), fam)
