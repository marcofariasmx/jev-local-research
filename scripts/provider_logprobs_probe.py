"""Qué modelos devuelven logprobs, probado directo contra OpenAI y Gemini."""
import json, urllib.error, urllib.request
import os
OA, GG = os.environ["OPENAI_API_KEY"], os.environ["GEMINI_API_KEY"]
P = "Is the sky blue on a clear day? Options: A) yes B) no. Answer with only the letter."

def post(url, body, h):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **h})
    try:
        with urllib.request.urlopen(req, timeout=60) as r: return 200, json.load(r)
    except urllib.error.HTTPError as e: return e.code, e.read().decode()[:220]

def show(name, s, r, f):
    if s != 200: print(f"[{name}] HTTP {s}: {r}"); return
    try: v = f(r)
    except Exception as e: v = f"(sin logprobs: {type(e).__name__})"
    print(f"[{name}] OK | {json.dumps(v)[:200]}")

oh = {"Authorization": f"Bearer {OA}"}
for m in ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.5-instant", "gpt-5.5"]:
    s, r = post("https://api.openai.com/v1/chat/completions", {"model": m, "messages": [{"role": "user", "content": P}],
        "max_completion_tokens": 16, "logprobs": True, "top_logprobs": 5}, oh)
    show(f"openai chat {m}", s, r, lambda r: [(t["token"], round(t["logprob"], 3)) for t in r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]])
    s, r = post("https://api.openai.com/v1/responses", {"model": m, "input": P, "max_output_tokens": 16,
        "top_logprobs": 5, "include": ["message.output_text.logprobs"]}, oh)
    show(f"openai responses {m}", s, r, lambda r: [[(t["token"], round(t["logprob"], 3)) for t in c["logprobs"][0]["top_logprobs"]] for o in r["output"] if o["type"] == "message" for c in o["content"]])
for m in ["gemini-2.5-flash", "gemini-2.5-flash-lite-001", "gemini-3.5-flash", "gemini-3.1-flash-lite"]:
    s, r = post(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent", {"contents": [{"role": "user", "parts": [{"text": P}]}],
        "generationConfig": {"maxOutputTokens": 16, "responseLogprobs": True, "logprobs": 5}}, {"x-goog-api-key": GG})
    show(f"gemini {m}", s, r, lambda r: [(c["token"], round(c["logProbability"], 3)) for c in r["candidates"][0]["logprobsResult"]["topCandidates"][0]["candidates"]])
