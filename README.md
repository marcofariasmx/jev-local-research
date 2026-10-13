# Local Jev: typed decisions with open LLMs

Research to answer two questions:
1. Can **Jev** (TypeSafe AI) be replicated locally, without an API key?
2. How well does it work, and how would it apply in production?

**Short answer: yes.** Jev's "trick" can be reproduced with any LLM that exposes `logprobs`. The best-known open-source library, **SemIf**, adds nothing over reading the logprobs directly. The best local result was **Qwen3.5-9B: 94% accuracy with good calibration, ~1 s per decision**.

---

## 1. What Jev is

- A model by TypeSafe AI, in early access. Closed-source, API-only, no open weights.
- **It is not a chat model.** It takes a context, a question and a fixed list of options, and returns a probability per option.
  - Example: "Which queue should this ticket go to?" → `support 0.94 · billing 0.04 · sales 0.02`
- Built for the **small decisions inside an agent**: routing, classification, approve/reject, judging evidence.
- Intended pattern: *the LLM writes, Jev decides, code acts*.

## 2. The trick

Before emitting each token, an LLM computes a probability distribution over its whole vocabulary. If the options are labeled with letters (A, B, C) and the model is told "answer with only the letter", **the probability of A, B and C at the first position is the decision**.

- **One forward pass:** no text generation, no JSON to parse.
- **Explicit probabilities:** enables threshold routing ("if confidence is below 0.9, escalate to a human").
- **Options defined at runtime:** no fine-tune per use case.
- In APIs this is called `logprobs` / `top_logprobs`.

## 3. What was compared

**Test set:** `authored144`, from the SemIf repo (MIT). 144 English decisions with known correct answers, in three families:
- `evidence_interpretation`: does the evidence support, contradict, or not settle the claim?
- `rule_application`: does the rule apply?
- `candidate_selection`: which candidate meets the criterion?

**Same prompt for every system:** SemIf's `direct_messages()`, with options as letters and "answer with only the letter".

| System | How the decision is read |
|---|---|
| **SemIf** + Qwen3.5-4B (MLX, 8-bit) | Native letter logits, one pass |
| **Docker Model Runner** + Qwen3.5-4B (IQ4) | First-token `top_logprobs` via the OpenAI-compatible API |
| **Docker Model Runner** + Qwen3.5-9B (Q4) | Same as above |
| **OpenAI gpt-5.4-nano** | First-token `logprobs` (OpenAI only returns top-1) |
| *Hosted Jev* | Not run (requires an API key). Its published reference number comes from a different benchmark. |

Hardware: Apple Silicon MacBook with 16 GB of RAM.

## 4. Results

| System | Accuracy | Balanced acc | ECE ↓ | Brier ↓ | NLL ↓ | p50 latency | RAM / cost |
|---|---|---|---|---|---|---|---|
| SemIf Qwen 4B | 81.3% | 0.814 | 0.053 | 0.244 | 0.423 | 0.76 s | ~4.5 GB |
| DMR Qwen 4B | 83.3% | 0.834 | 0.063 | 0.272 | 0.461 | 0.58 s | ~4–5 GB |
| **DMR Qwen 9B** | **93.8%** | **0.938** | **0.050** | **0.114** | **0.250** | 1.0 s | ~6.6 GB |
| gpt-5.4-nano | 83.3% | 0.834 | 0.096 | 0.267 | 0.576 | 0.67 s | API (~$0.005 for all 144) |

**Accuracy by family:**

| System | candidate | evidence | rule |
|---|---|---|---|
| SemIf 4B | 69% | 88% | 88% |
| DMR 4B | 75% | 92% | 83% |
| DMR 9B | 94% | 98% | 90% |
| nano | 92% | 88% | 71% |

**When the model decides only above a confidence threshold (and escalates the rest):**

| Threshold | nano: coverage / accuracy | 9B: coverage / accuracy |
|---|---|---|
| ≥ 0.90 | 79% / 90.4% | 74% / 96.3% |
| ≥ 0.99 | 59% / 96.5% | 36% / 98.1% |

**ECE** (calibration): when the model says "90% sure", is it right ~90% of the time? Lower is better.

### Takeaways
1. **SemIf adds nothing over raw logprobs.** With the same 4B model it scored the same or slightly lower. The value is in the trick, not the library.
2. **Model size is the dominant factor:** accuracy rises from 83% to 94% going from 4B to 9B.
3. **gpt-5.4-nano** matches the 4B on accuracy but is more overconfident. With a 0.99 threshold it handles ~60% of cases at 96.5% accuracy, at negligible cost.
4. SemIf reproduces its own published number (0.813 on authored144).

## 5. Which API models expose logprobs

Tested directly against each provider ([`scripts/provider_logprobs_probe.py`](scripts/provider_logprobs_probe.py)):

| Model | Logprobs |
|---|---|
| OpenAI `gpt-5.4-nano`, `gpt-5.4-mini` | ✅ Only the chosen token (top-1), even when asking for `top_logprobs: 5` |
| OpenAI `gpt-5.5` (reasoning) | ❌ "not supported with reasoning models" |
| Gemini `2.5-flash`, `3.5-flash`, `3.1-flash-lite` | ❌ "Logprobs is not enabled" |
| Anthropic Claude | ❌ Not offered by the API |
| Local Qwen (llama.cpp / DMR) | ✅ Full distribution (top-k) |

## 6. Local runtime configuration

Required settings for running the 4B/9B models on a 16 GB machine:

| Setting | Value | Reason |
|---|---|---|
| Context size | `--context-size 8192` | Native context (128K) raises memory to ~9.5 GB. Not persisted across `unload`; re-apply before each run. |
| llama.cpp prompt cache | `--cache-ram 0` | The default (8192 MiB) retains every distinct prompt in host memory. |
| SemIf precision | `--mlx-bits 8` | BF16 (~9 GB) does not fit alongside other workloads. Same decisions, ~0.75 s per row. |

```bash
docker model configure --context-size 8192 ai/qwen3.5:9b-q4_K_XL -- --cache-ram 0
```

[`scripts/run_guarded.sh`](scripts/run_guarded.sh) applies this configuration and runs the baseline under a memory watchdog (free RAM, `llama-server` RSS cap, swap growth). It does not retry and always unloads the model on exit.

## 7. How this applies in production

| Option | Pros | Cons |
|---|---|---|
| **Hosted Jev** | Most accurate and fastest to integrate | New vendor, data leaves your cloud, early access |
| **gpt-5.4-nano/mini + logprobs** | Available today, negligible cost per decision | Top-1 only, more overconfident |
| **Self-hosted Qwen 9B (vLLM)** | Best measured quality, full distribution, data stays in your cloud | You operate GPU infrastructure |
| **Large LLM with structured output (enum)** | Works with any provider | No probabilities, slower and more expensive |

**Recommended pattern:** the small model decides when its confidence is ≥ the threshold; otherwise it escalates to a large LLM or a human.

**Before deciding:**
- Re-run on **real data in your production language**. authored144 is synthetic and in English.
- Collect 200–300 labeled decisions from the actual use case.
- Pick the threshold by weighing the cost of an error against the cost of escalating.

---

## Reproduce

Requirements: Apple Silicon macOS, [uv](https://docs.astral.sh/uv/), Docker Desktop with Model Runner (`docker desktop enable model-runner --tcp=12434`).

```bash
./setup.sh                                   # clones SemIf (commit 23cf1f3) and installs its venv
export SEMIF_REPO=semif

# SemIf (downloads ~9 GB of weights the first time)
semif/.venv/bin/semif-score --backend mlx --mode direct --mlx-bits 8 \
  --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input semif/benchmarks/data/authored144.jsonl --output results/raw/semif-mlx8-authored144.jsonl

# Docker Model Runner with memory guards
docker model pull ai/qwen3.5:9b-q4_K_XL
MODEL=ai/qwen3.5:9b-q4_K_XL OUT=dmr9b-authored144.jsonl RSS_MAX_MB=8704 ./scripts/run_guarded.sh

# OpenAI
OPENAI_API_KEY=... semif/.venv/bin/python scripts/openai_baseline.py gpt-5.4-nano \
  semif/benchmarks/data/authored144.jsonl results/raw/nano-authored144.jsonl

# Comparison table
semif/.venv/bin/python scripts/compare.py
```

## Files

| Path | What it is |
|---|---|
| `scripts/dmr_baseline.py` | Typed decision via `top_logprobs` on any OpenAI-compatible endpoint (DMR), with RAM guards |
| `scripts/openai_baseline.py` | Same against OpenAI (top-1) |
| `scripts/run_guarded.sh` | Runner with a memory watchdog and `--cache-ram 0` |
| `scripts/compare.py` | Accuracy, balanced acc, NLL, Brier, ECE, latency and per-family breakdown |
| `scripts/provider_logprobs_probe.py` | Which OpenAI and Gemini models accept logprobs |
| `results/raw/*.jsonl` | Row-level predictions for each system |

**Data notes:**
- In `dmr9b-authored144.jsonl`, the first ~52 rows ran without the runtime configuration above; their latency is inflated. Predictions are unaffected.
- `semif-bf16-examples.jsonl` is a BF16 smoke test on the 3 bundled examples; latency is not representative.

## References
- SemIf (formerly OpenJev): https://github.com/TheoLeeCJ/SemIf (MIT)
- Jev integration with LangChain: https://www.langchain.com/blog/building-a-harness-with-jev
- Jev in DeepEval: https://deepeval.com/blog/introducing-jev-in-deepeval
