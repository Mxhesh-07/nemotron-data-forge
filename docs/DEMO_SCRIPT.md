# Demo video — script & shot list

**Target length: 2:45** (hard cap 3:00). Resolution 1080p, 30 or 60 fps.

**Premise (one line):** type a topic → Tavily finds *live* sources → NVIDIA
Nemotron on **Nebius Token Factory** writes, grades and de-duplicates a
fine-tuning dataset → download ready-to-train JSONL.

## What the video must prove (judging)

| Requirement | Where it is shown |
| --- | --- |
| Runs on Nebius Token Factory | Sidebar "Models used" shows the Token Factory base URL; `--check` output |
| Uses NVIDIA Nemotron | Generator = `NVIDIA-Nemotron-3-Super`, Grader = `NVIDIA-Nemotron-3-Ultra` on screen and in the reports |
| Real Tavily call (bonus) | Sources tab + `manifest.json` show real result URLs immediately after typing the topic |
| It's a product, not a script | Grading, rejection reasons, de-dupe counts, cost, manifest, export formats |
| Closes the loop | LoRA before/after improvement on the produced dataset |

## Pre-record checklist (do this first, takes ~3 min)

1. `.env` → `LLM_PROVIDER=nebius`, `NEBIUS_API_KEY=...`, `TAVILY_API_KEY=...`.
2. `python -m nemotron_data_forge --check` → every line `[PASS]`. **Do not film; just confirm.**
3. `streamlit run app.py`, browser at 100% zoom, notifications off, bookmarks bar hidden.
4. Terminal: 16pt+ font, dark theme, and never `cat .env` — keys must never appear.
5. Have these open in tabs: Streamlit app, `examples/tax-freelancers/quality-report.md`,
   `finetune_output/before_after.md` (from the LoRA run), the repo README.
6. Optional but strong: pre-warm with a throwaway topic so the first logged model call
   is instant, then reload the page before the real take.

## Shot list

| # | Time | Shot | On screen | Voiceover |
| --- | --- | --- | --- | --- |
| 1 | 0:00–0:10 | Title card | "Nemotron Data Forge — a topic in, a graded fine-tuning dataset out" | "Fine-tuning lives or dies by its data." |
| 2 | 0:10–0:22 | README pipeline diagram | `Topic → Tavily → Nemotron Super → Nemotron Ultra → dedupe → JSONL` | "This app turns any topic into a clean, graded, de-duplicated dataset — live." |
| 3 | 0:22–0:40 | Streamlit sidebar | Provider label "Nebius Token Factory", models list, pipeline knobs | "It runs on Nebius Token Factory and uses two NVIDIA Nemotron models: Super to write, Ultra to grade." |
| 4 | 0:40–0:55 | Type topic + Generate | Topic `Indian tax law for freelancers`, Standard settings, click Generate | "I'll type a niche topic and hit Generate." |
| 5 | 0:55–1:20 | Progress bar + Sources | Live Tavily sources appearing; progress steps | "First, Tavily pulls fresh, real pages from the web — no hard-coded knowledge." |
| 6 | 1:20–1:40 | Results header + Kept tab | Metrics: Kept / Rejected / Duplicates; open one kept example with scores | "Nemotron Ultra scored every example on accuracy, helpfulness and clarity, kept the good ones, and cleaned near-duplicates." |
| 7 | 1:40–1:55 | Rejected tab + Sources | Open a rejected example: reason + source URL | "Failures aren't hidden — each rejection keeps its reason and its source." |
| 8 | 1:55–2:08 | Downloads + run manifest | Download `dataset.jsonl`, `quality-report.md`, `manifest.json`; open manifest tab | "Everything is auditable: token usage, estimated cost, and a full run manifest." |
| 9 | 2:08–2:28 | `before_after.md` | Side-by-side base vs LoRA answers + offline metric deltas | "Then we fine-tune a small model on that dataset and measure the before/after." |
| 10 | 2:28–2:42 | Repo + tests | README, MIT license, green CI badge, `pytest` count | "Open source, MIT, fully tested, and reproducible end to end." |
| 11 | 2:42–2:45 | End card | Repo URL + "Try it" | "Nemotron Data Forge. Link below." |

## Full voiceover (read at a calm pace — ~2:45)

**[0:00]** Fine-tuning lives or dies by the data you feed it.

**[0:08]** Nemotron Data Forge turns any topic into a clean, graded,
de-duplicated dataset for fine-tuning — end to end, in one run.

**[0:16]** You give it a topic. Tavily retrieves live sources from the web.
NVIDIA Nemotron 3 Super writes grounded instruction and answer pairs from those
sources. Nemotron 3 Ultra then grades every example on accuracy, helpfulness and
clarity, and low scorers are rewritten or dropped. Near-duplicates are removed
with embedding similarity.

**[0:36]** It all runs on Nebius Token Factory through the OpenAI-compatible API.
Let me show it.

**[0:42]** I'll use a deliberately niche topic — Indian tax law for freelancers —
and keep the standard settings.

**[0:52]** Generate. Tavily goes out to the live web first, so the examples are
based on current, real pages — not the model's memory.

**[1:08]** Nemotron Super is writing now, and Ultra is grading each answer
against its source. You can watch progress and token usage as it happens.

**[1:24]** Here's the result: the grader kept the strong examples, and you can see
each one with its accuracy, helpfulness and clarity scores, right down to the
source link.

**[1:42]** Rejected examples are kept too, with the grader's reason — this is a
dataset you can audit, not a black box.

**[1:54]** Download the dataset as Alpaca, ShareGPT, OpenAI-chat or raw JSONL,
along with a quality report, a Hugging Face dataset card, and a run manifest
with per-model token usage and estimated cost.

**[2:08]** To close the loop, we fine-tune a small open model on the dataset this
app produced, using a LoRA adapter. Same weights, adapter off versus on, so the
only difference is the data. We score both with offline metrics — token-F1,
keyword recall, length and format adherence — so the effect of the data is
measurable, not vibes.

**[2:30]** The code is open source under MIT, with a CI pipeline that runs the
full test suite on every push, plus a hosted demo for judges.

**[2:42]** Nemotron Data Forge — type a topic, ship a dataset. Link below.

## Post-production

- Bold the on-screen metric numbers (Kept / Rejected / Duplicates / cost).
- Caption key model IDs once: `NVIDIA-Nemotron-3-Super-120b-a12b`,
  `NVIDIA-Nemotron-3-Ultra-550b-a55b`.
- Blur/omit any API key field contents. Never show the AIza / nvapi / tvly strings.
- Keep it under 3:00 — if long, shorten shot 2 and read the voiceover faster.

## Fallback if the Nebius key is still pending on recording day

Do **not** film the NVIDIA NIM fallback as the submission demo — the rules
require the Token Factory run. Instead:

1. Post the organizer request (`docs/ORGANIZER_REQUEST.md`) immediately.
2. Film shots 1, 2, 9, 10 and 11 (they don't need a live run).
3. Record shots 3–8 the moment a Token Factory key arrives, then stitch.

