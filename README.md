# Jev local: decisiones tipadas con LLMs abiertos

Research (octubre 2026) para responder dos preguntas:
1. ¿Se puede replicar **Jev** (TypeSafe AI) localmente, sin API key?
2. ¿Qué tan bien funciona y cómo aplicaría en producción?

**Respuesta corta:** sí. El "truco" de Jev se reproduce con cualquier LLM que exponga `logprobs`. La librería abierta más conocida, **SemIf**, no aporta nada sobre leer los logprobs directo. El mejor resultado local fue **Qwen3.5-9B: 94% de aciertos con buena calibración, ~1 s por decisión**.

---

## 1. Qué es Jev

- Es un modelo de TypeSafe AI, lanzado en septiembre de 2026. Es cerrado, se usa solo por API y no tiene pesos abiertos.
- **No es un chatbot.** Recibe un contexto, una pregunta y una lista fija de opciones, y devuelve una probabilidad por opción.
  - Ejemplo: "¿A qué cola va este ticket?" → `soporte 0.94 · billing 0.04 · ventas 0.02`
- Sirve para las **decisiones chicas dentro de un agente**: rutear, clasificar, aprobar o rechazar, juzgar evidencia.
- La propuesta: *el LLM escribe, Jev decide, el código actúa*.

## 2. El truco

Antes de escribir cada token, un LLM calcula la probabilidad de todos los tokens posibles. Si le das las opciones con letras (A, B, C) y le pides "responde solo la letra", basta con **leer la probabilidad de A, B y C en el primer token**.

- Es **una sola pasada**: no genera texto ni hay JSON que parsear.
- Da **probabilidades**, no solo la respuesta, así que puedes decidir con un umbral ("si la confianza baja de 0.9, escala a un humano").
- En las APIs esto se llama `logprobs` / `top_logprobs`.

## 3. Qué se comparó

**Set de prueba:** `authored144`, del repo de SemIf (MIT). Son 144 decisiones en inglés con respuesta correcta conocida, en tres familias:
- `evidence_interpretation`: ¿la evidencia soporta, contradice o no alcanza?
- `rule_application`: ¿aplica la regla?
- `candidate_selection`: ¿cuál candidato cumple?

**Mismo prompt para todos:** `direct_messages()` de SemIf, con las opciones en letras y "responde solo la letra".

| Sistema | Cómo lee la decisión |
|---|---|
| **SemIf** + Qwen3.5-4B (MLX, 8-bit) | Logits nativos de las letras, una pasada |
| **Docker Model Runner** + Qwen3.5-4B (IQ4) | `top_logprobs` del primer token vía API OpenAI-compatible |
| **Docker Model Runner** + Qwen3.5-9B (Q4) | Igual que el anterior |
| **OpenAI gpt-5.4-nano** | `logprobs` del primer token (OpenAI solo da el top-1) |
| *Jev hospedado* | No corrido (requiere API key). Su referencia publicada es de otro benchmark. |

Hardware: MacBook Apple Silicon con 16 GB de RAM.

## 4. Resultados

| Sistema | Aciertos | Balanced acc | ECE ↓ | Brier ↓ | NLL ↓ | Latencia p50 | RAM |
|---|---|---|---|---|---|---|---|
| SemIf Qwen 4B | 81.3% | 0.814 | 0.053 | 0.244 | 0.423 | 0.76 s | ~4.5 GB |
| DMR Qwen 4B | 83.3% | 0.834 | 0.063 | 0.272 | 0.461 | 0.58 s | ~4–5 GB |
| **DMR Qwen 9B** | **93.8%** | **0.938** | **0.050** | **0.114** | **0.250** | 1.0 s | ~6.6 GB |
| gpt-5.4-nano | 83.3% | 0.834 | 0.096 | 0.267 | 0.576 | 0.67 s | API (~$0.005 por las 144) |

**Por familia (aciertos):**

| Sistema | candidate | evidence | rule |
|---|---|---|---|
| SemIf 4B | 69% | 88% | 88% |
| DMR 4B | 75% | 92% | 83% |
| DMR 9B | 94% | 98% | 90% |
| nano | 92% | 88% | 71% |

**Si el modelo decide solo cuando su confianza pasa un umbral (y escala el resto):**

| Umbral | nano: cubre / acierta | 9B: cubre / acierta |
|---|---|---|
| ≥ 0.90 | 79% / 90.4% | 74% / 96.3% |
| ≥ 0.99 | 59% / 96.5% | 36% / 98.1% |

**ECE** (calibración): cuando el modelo dice "90% seguro", ¿acierta ~90% de las veces? Más bajo es mejor.

### Conclusiones
1. **SemIf no aporta sobre logprobs crudos.** Con el mismo modelo 4B dio igual o peor. El valor está en el truco, no en la librería.
2. **El tamaño del modelo es lo que manda:** de 83% a 94% al pasar de 4B a 9B.
3. **gpt-5.4-nano** iguala al 4B en aciertos pero está más sobreconfiado. Con un umbral de 0.99 resuelve ~60% de los casos con 96.5% de acierto, por centavos.
4. SemIf reproduce su número publicado (0.813 en authored144).

## 5. Qué modelos de API exponen logprobs

Probado directo contra cada proveedor el 2026-10-12 ([`scripts/provider_logprobs_probe.py`](scripts/provider_logprobs_probe.py)):

| Modelo | Logprobs |
|---|---|
| OpenAI `gpt-5.4-nano`, `gpt-5.4-mini` | ✅ Solo el token elegido (top-1), aunque pidas `top_logprobs: 5` |
| OpenAI `gpt-5.5` (razonador) | ❌ "not supported with reasoning models" |
| Gemini `2.5-flash`, `3.5-flash`, `3.1-flash-lite` | ❌ "Logprobs is not enabled" |
| Anthropic Claude | ❌ La API no los ofrece |
| Qwen local (llama.cpp / DMR) | ✅ Distribución completa (top-k) |

### Gateways corporativos de LLMs
Si consumes los modelos a través de un gateway (KrakenD u otro) que **rearma el body del request con una lista blanca de campos** (típico cuando hay DLP o redacción de PII), revisa que `logprobs` y `top_logprobs` estén en esa lista. En el gateway que revisé:
- La ruta **Chat Completions** los descartaba en silencio (fix: ~10 líneas en la plantilla).
- La ruta **Responses** sí reenviaba `top_logprobs` e `include`.
- La respuesta pasaba completa en ambas.

## 6. Lecciones de RAM (Mac de 16 GB)

Correr esto localmente **tumbó la máquina la primera vez** (19 GB de swap). Hubo dos causas, ninguna obvia:

1. **`docker model configure --context-size` no persiste tras un `unload`.** La siguiente carga vuelve al contexto nativo (128K) y el 4B o el 9B llegan a ~9.5 GB. Hay que reaplicarlo justo antes de usar el modelo.
2. **llama.cpp trae `--cache-ram 8192` por default.** Guarda cada prompt distinto en RAM, el sistema lo manda a swap y **el RSS se ve plano** mientras el swap crece. Es invisible en Activity Monitor.

**Configuración que funcionó:**
```bash
docker model configure --context-size 8192 ai/qwen3.5:9b-q4_K_XL -- --cache-ram 0
```

Además:
- **SemIf en BF16** (~9 GB) no cabe junto con otras apps: tarda ~90 s por fila por swap. Con `--mlx-bits 8` baja a ~0.75 s y da las mismas decisiones.
- [`scripts/run_guarded.sh`](scripts/run_guarded.sh) corre el baseline con watchdog: aborta si la RAM libre baja de 15%, si `llama-server` pasa de un tope o si el swap crece de golpe. **Nunca reintenta** y siempre descarga el modelo al final.

## 7. Cómo aplicaría en producción

| Opción | Pros | Contras |
|---|---|---|
| **Jev hospedado** | Lo más preciso y rápido de integrar | Proveedor nuevo, datos salen de tu nube, early access |
| **gpt-5.4-nano/mini + logprobs** | Ya disponible, centavos por decisión | Solo top-1, más sobreconfiado; revisar que el gateway no descarte el campo |
| **Qwen 9B hospedado (vLLM)** | Mejor calidad medida, distribución completa, datos en tu nube | Operar infraestructura con GPU |
| **LLM grande con salida estructurada (enum)** | Funciona con cualquier proveedor | Sin probabilidades, más lento y caro |

**Patrón recomendado:** el modelo chico decide si su confianza ≥ umbral; si no, escala a un LLM grande o a un humano.

**Antes de decidir:**
- Repetir con **datos reales en el idioma de producción**. authored144 es sintético y en inglés.
- Juntar 200–300 decisiones etiquetadas del caso de uso.
- Elegir el umbral según el costo de un error contra el costo de escalar.

---

## Reproducir

Requisitos: macOS Apple Silicon, [uv](https://docs.astral.sh/uv/), Docker Desktop con Model Runner (`docker desktop enable model-runner --tcp=12434`).

```bash
./setup.sh                                   # clona SemIf (commit 23cf1f3) e instala su venv
export SEMIF_REPO=semif

# SemIf (descarga ~9 GB de pesos la primera vez)
semif/.venv/bin/semif-score --backend mlx --mode direct --mlx-bits 8 \
  --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input semif/benchmarks/data/authored144.jsonl --output results/raw/semif-mlx8-authored144.jsonl

# Docker Model Runner con guardas de memoria
docker model pull ai/qwen3.5:9b-q4_K_XL
MODEL=ai/qwen3.5:9b-q4_K_XL OUT=dmr9b-authored144.jsonl RSS_MAX_MB=8704 ./scripts/run_guarded.sh

# OpenAI
OPENAI_API_KEY=... semif/.venv/bin/python scripts/openai_baseline.py gpt-5.4-nano \
  semif/benchmarks/data/authored144.jsonl results/raw/nano-authored144.jsonl

# Tabla comparativa
semif/.venv/bin/python scripts/compare.py
```

## Archivos

| Ruta | Qué es |
|---|---|
| `scripts/dmr_baseline.py` | Decisión tipada vía `top_logprobs` en cualquier endpoint OpenAI-compatible (DMR), con guardas de RAM |
| `scripts/openai_baseline.py` | Lo mismo contra OpenAI (top-1) |
| `scripts/run_guarded.sh` | Runner con watchdog de memoria y `--cache-ram 0` |
| `scripts/compare.py` | Accuracy, balanced acc, NLL, Brier, ECE, latencia y desglose por familia |
| `scripts/provider_logprobs_probe.py` | Qué modelos de OpenAI y Gemini aceptan logprobs |
| `results/raw/*.jsonl` | Predicciones fila por fila de cada sistema |

**Notas sobre los datos:**
- En `dmr9b-authored144.jsonl`, las primeras ~52 filas se corrieron antes del fix de RAM (con swap), así que su latencia está inflada. Las predicciones no cambian.
- `semif-bf16-examples.jsonl` es el smoke-test en BF16 (~90 s por fila por swap).

## Referencias
- SemIf (antes OpenJev): https://github.com/TheoLeeCJ/SemIf (MIT)
- Integración de Jev con LangChain: https://www.langchain.com/blog/building-a-harness-with-jev
- Jev en DeepEval: https://deepeval.com/blog/introducing-jev-in-deepeval
