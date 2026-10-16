# App screenshots (DEV ONLY — NVIDIA NIM fallback)

These are **development screenshots** captured against the NVIDIA NIM fallback
endpoint. They are committed to prove the UI works end to end and to validate the
demo shot list. They are **NOT** the submission screenshots — the hackathon rules
require the graded demo/run to be on **Nebius Token Factory**, so the final
screenshots and video must be re-taken once a Token Factory key is provisioned.

Captured 2026-17-10 with a headless Chromium (Playwright) driving the real
Streamlit app on a live run (topic: *Indian tax law for freelancers*).

| File | What it shows |
| --- | --- |
| `01_home.png` | App home — sidebar (API keys, pipeline knobs), topic input, Generate button |
| `02_sidebar_models.png` | Sidebar "Models used" expander — generator / grader / embedding IDs |
| `03_running.png` | Mid-run — progress bar + live Tavily sources appearing |
| `04_results.png` | Results — success banner, Kept / Not kept / Duplicates / cost metrics, Kept tab |
| `04b_kept_example.png` | A kept example expanded — response + Accuracy / Helpfulness / Clarity scores + source link |
| `05_sources.png` | Sources tab — real Tavily result URLs with relevance scores |
| `06_manifest.png` | Run manifest — full JSON (config, models, usage, cost, sources) |
| `07_not_kept.png` | Not-kept tab — rejected example with the grader's reason |

## Provider note

The sidebar reads "NVIDIA NIM (development fallback)" because the NIM grader
model `nvidia/nemotron-3-ultra-550b-a55b` started returning HTTP 404 on the NIM
endpoint (most large NIM models are currently listed-but-404). The dev fallback
now uses `nvidia/nemotron-3-super-120b-a12b` for both generation and grading —
the only reliably callable Nemotron chat model on NIM right now. This does **not**
affect the graded run, which uses Nebius Token Factory's own model IDs.
