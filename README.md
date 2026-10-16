# 🧪 Nemotron Data Forge

**Type a topic → get a clean, graded, de-duplicated, fine-tuning-ready dataset.**

Nemotron Data Forge is a self-running workflow that turns any topic into a
high-quality instruction dataset for fine-tuning:

```
Topic → Tavily finds fresh sources → Nemotron 3 Super generates examples
      → Nemotron 3 Ultra grades & filters them → semantic dedupe → JSONL + report
```

Built for the **Nebius × NVIDIA Global AI Hackathon — Best Apps & Agents track**,
targeting the **Best Use of Tavily** bonus.

---

## Why it's more than a demo

Most "generate a dataset" scripts dump raw model output into a file. This one
treats dataset quality as the product:

- **Graded, not guessed.** Every example is scored on *accuracy*, *helpfulness*
  and *clarity* by a stronger model, then filtered by an average threshold plus
  a per-axis floor.
- **De-duplicated.** Near-duplicates are removed with embedding cosine
  similarity (falls back to token-set Jaccard if embeddings are unavailable).
- **Traceable.** Each example keeps its `source_url`, `source_title`, scores and
  the model IDs used, so anything can be audited back to a real web source.
- **Resilient.** Per-source and per-batch failures are captured and reported;
  one bad source never kills a run. Transient API errors are retried with
  exponential backoff and jitter.
- **Observable.** Every run writes a manifest with per-model token usage, an
  estimated USD cost, timings, and a list of any warnings.
- **Tested.** 148 unit/integration tests run entirely offline against fakes, plus
  a render test for the Streamlit UI and a CI workflow.

---

## How the required tools are used

| Requirement | Where it is used | File |
| --- | --- | --- |
| **Runs on Nebius Token Factory** | All model calls use the OpenAI-compatible Token Factory API at `https://api.tokenfactory.nebius.com/v1/` | `nemotron_data_forge/llm.py` |
| **NVIDIA Nemotron (generator)** | `nvidia/Nemotron-3-Super-120b-a12b` writes grounded instruction/response pairs | `nemotron_data_forge/generator.py` |
| **NVIDIA Nemotron (grader)** | `nvidia/Nemotron-3-Ultra-550b-a55b` scores and filters each example | `nemotron_data_forge/grader.py` |
| **Tavily (Best Use of Tavily bonus)** | Real runtime call to the Tavily Search API for fresh grounding sources | `nemotron_data_forge/search.py` |
| **Nebius Serverless Jobs** (scale) | Run the same CLI as a background job to build large datasets | `scripts/serverless_job.py`, `Dockerfile` |

> Why Nemotron 3 Super **and** Ultra? Super is fast and cheap for generation;
> Ultra is reserved for the harder judging step. That split keeps the app
> responsive and stretches credits — exactly what the track calls for.

---

## Quickstart

```bash
# 1. Install
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt

# 2. Configure keys
copy .env.example .env         # Windows  (cp on macOS/Linux)
# then edit .env and set NEBIUS_API_KEY and TAVILY_API_KEY

# 3. Run the demo UI
streamlit run app.py
```

Open the app, type a topic (e.g. *"Indian tax law for freelancers"*), pick a
format, and click **Generate dataset**. Download `dataset.jsonl`, the
`quality-report.md`, and the run `manifest.json`.

### API keys

- **NEBIUS_API_KEY** — from <https://tokenfactory.nebius.com> (Get API key).
- **TAVILY_API_KEY** — from <https://tavily.com>. The **Nebius Builder Program**
  (<https://dev.nebius.com/builders>) includes $25 of Tavily credit.
- No Tavily key yet? Tick **Demo mode** in the sidebar or pass `--demo` to use
  three bundled sample sources instead of live search — you still need a provider
  key (Nebius / NVIDIA / OpenRouter) for generation and grading. The Tavily bonus
  requires running in normal mode with a real key.

#### Development fallback providers (not for submission)

Nebius Token Factory requires a bank card to finish onboarding, which can block
development. For local testing only you can point the same OpenAI-compatible
client at one of two card-free endpoints:

**Option A — NVIDIA NIM** (<https://build.nvidia.com>, key `NVIDIA_API_KEY`, no
card) runs the real Nemotron 3 models through the same OpenAI-compatible client.
Note: NIM currently lists many models but most return HTTP 404, so the fallback
uses `nvidia/nemotron-3-super-120b-a12b` for both generation and grading (the only
reliably callable Nemotron chat model there right now):

```dotenv
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...
```

**Option B — OpenRouter** (<https://openrouter.ai>, key `OPENROUTER_API_KEY`, no
card) serves Nemotron 3 over an OpenAI-compatible API. Append `:free` to a model
ID (e.g. `nvidia/nemotron-3-super-120b-a12b:free`) for the zero-price route —
those routes log prompts and have no SLA, so prefer them only for throwaway dev
runs.

```dotenv
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
```

Both are **development conveniences only** — the hackathon submission must run
on Nebius Token Factory, so the recorded demo and the graded run use the default
`LLM_PROVIDER=nebius`. Switching providers is a one-line change; model IDs and
the base URL are configurable per provider via `config.py` presets and the
`GENERATOR_MODEL`, `GRADER_MODEL`, `FAST_MODEL`, `EMBEDDING_MODEL` and
`NEBIUS_BASE_URL` environment variables.

Once your keys are in `.env`, verify everything end to end with a few cheap live
calls before running a full job:

```bash
python -m nemotron_data_forge --check
# [PASS] NEBIUS_API_KEY present
# [PASS] List models - 42 available
# [PASS] generator model available - nvidia/Nemotron-3-Super-120b-a12b
# ...
# [PASS] Tavily search round-trip - 1 result(s)
```

---

## Command line (headless / for Serverless Jobs)

```bash
# Entry point is installed as `dataforge`, or use `python -m nemotron_data_forge`.
python -m nemotron_data_forge --check              # verify keys, models and connectivity
python -m nemotron_data_forge --topic "Indian tax law for freelancers" --format Q&A
python -m nemotron_data_forge --topic "intro to vector databases" --demo
python -m nemotron_data_forge --topic "..." --examples-per-source 4 --max-sources 6 --threshold 7.5
python -m nemotron_data_forge --topic "..." --export-format sharegpt --val-ratio 0.1   # ShareGPT + val split
python -m nemotron_data_forge --topic "..." --push-hf yourname/tax-freelancers-in   # optional HF push
python -m nemotron_data_forge --topic "..." --manifest-json                         # machine-readable run manifest
```

Key flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--check` | off | Verify keys, model access and connectivity, then exit |
| `--topic` | config default | Topic to build the dataset for |
| `--format` | `Q&A` | `Q&A`, `Chat`, or `Reasoning` |
| `--examples-per-source` | `3` | Examples requested per source |
| `--max-sources` | `4` | Maximum Tavily results to use |
| `--threshold` | `7.0` | Minimum average score to keep an example |
| `--min-axis-score` | `5.0` | Minimum score on every axis |
| `--concurrency` | `4` | Parallel model requests |
| `--rewrite-rounds` | `1` | Rewrite-on-reject rounds (0 disables) |
| `--max-usd` | `0` | Soft spend cap in USD for the run (0 = unlimited) |
| `--export-format` | `alpaca` | Record layout: `alpaca`, `sharegpt`, `openai`, `raw` |
| `--val-ratio` | `0` | Fraction held out as a validation split (e.g. `0.1`) |
| `--demo` | off | Use bundled sample sources (no Tavily) |
| `--no-embeddings` | off | Disable semantic de-duplication |
| `--out` | `outputs/` | Output directory |
| `--push-hf` | off | Push the dataset to a Hugging Face dataset repo |
| `--manifest-json` | off | Print the run manifest as JSON to stdout |

Outputs land in `outputs/` as `dataset-<stamp>.jsonl` (plus `val-<stamp>.jsonl`
when `--val-ratio` is set), `quality-report-<stamp>.md`, `manifest-<stamp>.json`
and `dataset-card-<stamp>.md`.

---

## What you get

- **`dataset.jsonl`** — instruction/response records ready for fine-tuning, each
  with its `source_url`, scores and topic for traceability. Choose the record
  layout with `--export-format`: **alpaca** (default), **sharegpt**, **openai**
  chat, or **raw**. `samples/example-dataset.jsonl` shows the alpaca format.
- **`quality-report.md`** — counts, average scores, per-model token usage,
  sources, and kept vs dropped examples (including near-duplicates) with the
  reason each was dropped.
- **`manifest.json`** — machine-readable record of the run: config, models,
  token usage, estimated cost, duration, sources and warnings.
- **`dataset-card.md`** — a Hugging Face-ready dataset card (YAML front matter).

---

## Architecture

```
app.py  /  nemotron_data_forge/cli.py      UI + headless entrypoints
        │
        ▼
nemotron_data_forge/pipeline.py            orchestration, concurrency, manifest
  ├── search.py       Tavily search → Source objects (retries)
  ├── generator.py    Nemotron 3 Super → Example objects (chunked, validated)
  ├── grader.py       Nemotron 3 Ultra → scores + threshold filter (batched)
  ├── dedupe.py       embedding-cosine or token-Jaccard de-duplication
  ├── export.py       JSONL + Markdown report + manifest (+ optional HF push)
  ├── llm.py          Token Factory client: retries, JSON repair, usage
  ├── validation.py   length/emptiness checks and score clamping
  ├── usage.py        thread-safe token/cost accounting
  ├── doctor.py       `--check`: keys, model access and connectivity
  └── config.py       env-driven settings + validation
```

---

## Production features

| Concern | How it's handled |
| --- | --- |
| Transient API failures | Retries with exponential backoff + jitter on 429/5xx/timeouts (`llm.py`, `search.py`) |
| Model returns bad JSON | JSON mode, graceful fallback, then a single repair turn; fenced/prose JSON is salvaged (`extract_json`) |
| Long sources / long batches | Sources are chunked; grading is batched (`GRADE_BATCH_SIZE`) |
| Rate limits & latency | Bounded concurrency with a thread pool (`--concurrency`) |
| Hallucination | Grader scores *accuracy against the source*; low scores are dropped or rewritten |
| Low-quality output | Rewrite-on-reject regenerates failed examples using the grader's reason (`--rewrite-rounds`) |
| Bad output | Length/emptiness validation, score clamping, per-axis floor |
| One source failing | Recorded per-source, run continues, surfaced in the manifest |
| Cost visibility | Per-model token accounting + estimated USD cost from a pricing table |
| Spend guardrail | Soft USD cap per run (`--max-usd`): stops new sources and skips optional phases once reached |
| Secrets | Keys come from env/`.env`, are never logged, and `.env` is git-ignored |

---

## Before/after fine-tuning demo

To prove the dataset is actually useful, `finetune/` fine-tunes a small base
model on it and compares answers **before** vs **after** — on a free Colab T4,
with **no API keys**. See [`finetune/README.md`](finetune/README.md).

```bash
# Validate any dataset with no heavy dependencies:
python -m finetune.train_lora --dataset finetune/sample_dataset.jsonl --check

# Train LoRA + generate the before/after report:
python -m finetune.train_lora --dataset outputs/dataset-<stamp>.jsonl \
  --base-model Qwen/Qwen2.5-0.5B-Instruct --epochs 3 --out finetune_output
```

It writes `finetune_output/before_after.md` (side-by-side answers plus offline
metrics: token-F1 vs reference, keyword recall, length closeness, format
adherence). The bundled `finetune/sample_dataset.jsonl` is a clearly-labelled
Python-fundamentals fixture used only to exercise the notebook; the real demo
uses the generated `outputs/dataset-*.jsonl`. Open
`finetune/colab_before_after.ipynb` in Colab to run it end to end.

---

## Testing & CI

```bash
pip install -e ".[dev]"
pytest          # 148 tests, fully offline (fakes mock the models and search)
ruff check .    # lint
```

Tests cover JSON extraction and retry logic, generation and grading (including
batching and score clamping), de-duplication, Tavily parsing/retries, the
end-to-end pipeline, exports, the CLI, and the Streamlit app render.
`.github/workflows/ci.yml` runs lint + tests on Python 3.10/3.11/3.12.

---

## Hosted demo (for judges)

The Streamlit app reads keys from host secrets, so it can be hosted for free for
the judging window. See [`DEPLOY.md`](DEPLOY.md) for one-step deploys to
**Streamlit Community Cloud** or **Hugging Face Spaces**, including the `st.secrets`
keys to set and a `DEFAULT_MAX_USD` spend cap for a public demo.

---

## Running at scale on Nebius Serverless Jobs

A `Dockerfile` is included so the same pipeline can run as a background job
(Serverless Jobs, Serverless Endpoints or DevPods all count as "runs on Nebius"):

```bash
docker build -t nemotron-data-forge .
docker run --rm -e NEBIUS_API_KEY=... -e TAVILY_API_KEY=... \
  -v "$PWD/outputs:/outputs" nemotron-data-forge \
  --topic "Indian tax law for freelancers" --examples-per-source 5 --max-sources 8 --out /outputs
```

Point the job at `scripts/serverless_job.py` (the image's entrypoint) and pass
the topic as an argument; keys are provided as environment variables. See
<https://docs.nebius.com/serverless/quickstart/jobs>.

---

## Models

| Role | Model ID | Input / Output (per 1M tokens) |
| --- | --- | --- |
| Generator | `nvidia/Nemotron-3-Super-120b-a12b` | $0.30 / $0.90 |
| Grader | `nvidia/Nemotron-3-Ultra-550b-a55b` | $1.00 / $3.00 |
| Fast (optional) | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | $0.06 / $0.24 |
| Embeddings | `Qwen/Qwen3-Embedding-8B` | $0.01 |

Model IDs are current as of Oct 2026 and can be overridden in `.env`.

---

## Configuration

All settings are environment variables (see `.env.example`): model IDs and the
base URL, pipeline defaults (topic, format, counts, thresholds), and limits
(`MAX_CONCURRENCY`, `MAX_RETRIES`, `REQUEST_TIMEOUT`, `GRADE_BATCH_SIZE`,
`MAX_OUTPUT_TOKENS`, `LLM_PROVIDER`, `LLM_DISABLE_THINKING`, `SOURCE_CHUNK_CHARS`,
`LOG_LEVEL`, `OUTPUT_DIR`, `DEFAULT_MAX_USD`).

---

## Roadmap

- [x] Before/after demo: fine-tune a small model on the dataset to prove quality
- [x] Rewrite step: re-generate examples the grader rejects (instead of dropping)
- [ ] Stream progress from a Serverless Job back to the UI

## License

MIT — see [LICENSE](LICENSE).
