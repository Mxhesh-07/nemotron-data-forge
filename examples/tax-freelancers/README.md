# Example run — Indian tax law for freelancers

A real, unedited output of the pipeline, committed so reviewers can inspect a
dataset without running anything. Every row is traceable back to a live web
source that Tavily retrieved.

## Files

| File | What it is |
| --- | --- |
| `dataset.jsonl` | 11 training rows (Alpaca layout, `instruction`/`input`/`output`) |
| `val.jsonl` | 1 held-out validation row (10% split) |
| `quality-report.md` | Grader scores, kept/rejected examples, token usage, source list |
| `manifest.json` | Full run manifest: config, models, usage, cost, sources, errors |
| `dataset-card.md` | Hugging Face dataset card for the run |

## How it was produced

| Setting | Value |
| --- | --- |
| Topic | `Indian tax law for freelancers` |
| Format | `Q&A` |
| Provider | NVIDIA NIM (development fallback) |
| Generator | `nvidia/nemotron-3-super-120b-a12b` |
| Grader | `nvidia/nemotron-3-ultra-550b-a55b` |
| Embeddings | `nvidia/nemotron-3-embed-1b` |
| Sources | 4, retrieved live by Tavily |
| Generated / graded / kept | 12 / 12 / 12 |
| Duplicates removed | 0 |
| Avg accuracy / helpfulness / clarity | 10.0 / 9.0 / 9.0 |
| Total tokens | 6,488 |
| Duration | 14.5 s |

### A note on the provider

This example was generated against the **NVIDIA NIM development endpoint**, which
runs the same NVIDIA Nemotron 3 models through the same OpenAI-compatible client
the app uses for Nebius Token Factory. It is a card-free development fallback so
the pipeline can be exercised before Token Factory access is provisioned; the
recorded submission run and the hosted demo use `LLM_PROVIDER=nebius`
(`https://api.tokenfactory.nebius.com/v1/`) and the Token Factory model IDs
(`nvidia/Nemotron-3-Super-120b-a12b`, `nvidia/Nemotron-3-Ultra-550b-a55b`).

## Reproduce it

With a Nebius Token Factory key and a Tavily key in `.env`:

```bash
dataforge \
  --topic "Indian tax law for freelancers" \
  --format "Q&A" \
  --examples-per-source 3 \
  --max-sources 4 \
  --threshold 7.0 \
  --export-format alpaca \
  --val-ratio 0.1 \
  --out examples/tax-freelancers
```

Or open the Streamlit app (`streamlit run app.py`), enter the topic, choose
**Standard**, and download the result.

