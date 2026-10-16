# Hackathon submission — paste-ready package

Everything below is drafted and ready to copy into the Devpost submission form.
Items marked `[FILL AFTER RUN]` depend on the live Nebius Token Factory run and
must be completed once the key lands — do not submit them as final before then.

- **Track:** Best Apps & Agents
- **Bonus target:** Best Use of Tavily
- **Repo:** https://github.com/Mxhesh-07/nemotron-data-forge
- **Demo:** `[FILL AFTER DEPLOY]`
- **Video:** `[FILL AFTER RECORDING]`
- **Submission deadline:** Oct 30, 2026, 10:00 AM PT

---

## Title

**Nemotron Data Forge — turn any topic into a graded, de-duplicated fine-tuning dataset**

## Tagline (≤ 140 chars)

Type a topic. Get a clean, graded, traceable instruction dataset ready for fine-tuning — powered by Tavily + NVIDIA Nemotron on Nebius Token Factory.

## One-liner

A self-running pipeline that turns any topic into a high-quality instruction dataset: Tavily finds fresh sources, Nemotron 3 Super generates examples, Nemotron 3 Ultra grades and filters them, and embeddings remove near-duplicates.

## Long description (Devpost "What it does" / "How we built it")

Most "generate a dataset" scripts dump raw model output into a file. Nemotron
Data Forge treats dataset quality as the product. Give it a topic and it runs a
full, auditable pipeline:

```
Topic → Tavily finds fresh web sources → Nemotron 3 Super generates examples
      → Nemotron 3 Ultra grades & filters (accuracy / helpfulness / clarity)
      → embedding dedupe → JSONL + quality report + manifest
```

**Why it's more than a demo**

- **Graded, not guessed.** Every example is scored on *accuracy*, *helpfulness*
  and *clarity* by a stronger model, then filtered by an average threshold plus a
  per-axis floor. Rejected examples get one automatic rewrite attempt.
- **De-duplicated.** Near-duplicates are removed with embedding cosine
  similarity (falls back to token-set Jaccard if embeddings are unavailable).
- **Traceable.** Each row keeps its `source_url`, `source_title`, grader scores and
  the model IDs used, so anything can be audited back to a real web source.
- **Resilient.** Per-source and per-batch failures are captured and reported; one
  bad source never kills a run. Transient API errors retry with exponential
  backoff and jitter.
- **Observable.** Every run writes a manifest with per-model token usage, an
  estimated USD cost, timings, and any warnings.
- **Open source.** 148 offline tests, a Streamlit app, a CLI, and a Docker image.

**Tech stack**

- **Nebius Token Factory** — OpenAI-compatible inference API for generation,
  grading and embeddings.
- **Tavily** — real-time web search for grounding sources (Best Use of Tavily bonus).
- **NVIDIA Nemotron 3** — Super (generation) and Ultra (grading) models.
- **Python** (`openai` SDK), **Streamlit** app, **Docker**, **GitHub Actions** CI.

**Example output**

A committed, unedited run on *"Indian tax law for freelancers"* is in
[`examples/tax-freelancers/`](../examples/tax-freelancers): 12 kept examples,
avg 10.0 / 9.0 / 9.0 (accuracy / helpfulness / clarity), ~6.5k tokens, estimated
cost **$0.006**.

---

## Required feedback (rules: feedback on Nebius tools + NVIDIA models is mandatory)

> Complete the `[FILL]` parts from the live Token Factory run before submitting.

**Nebius Token Factory**
- What I used it for: `[FILL — generation, grading, embeddings; number of calls, latency]`
- What worked well: `[FILL]`
- What needs work: `[FILL]`
- Onboarding (zero → hello world): `[FILL — the bank-card billing wall for solo
  participants; did the hackathon promo credit / Builders Program resolve it?]`
- Would I build with it again, and why: `[FILL]`

**NVIDIA models (Nemotron 3 Super / Ultra)**
- What I used them for: Super = generation, Ultra = grading.
- What worked well: `[FILL — JSON validity, instruction quality, grading consistency]`
- What needs work: `[FILL — reasoning-token budget, truncation, latency]`
- Would I build with them again, and why: `[FILL]`

**Nebius AI Cloud** (if used)
- `[FILL or "not used — inference ran entirely on Token Factory"]`

---

## What's next

- [ ] Live graded run on **Nebius Token Factory** and re-record the demo video.
- [ ] Deploy the public demo (Streamlit Community Cloud) with a spend cap.
- [ ] Larger multi-topic runs and a LoRA fine-tune to show downstream value.
- [ ] Publish the example datasets to the Hugging Face Hub.

## Submission checklist

- [ ] Repo public with README, LICENSE, and a reproducible example.
- [ ] Live demo URL (judges can use it free).
- [ ] Demo video (≤ 3 min) showing a real end-to-end run.
- [ ] Screenshots of the app + a sample of the generated dataset.
      (Dev shots in `shots/` prove the UI; **re-take on Token Factory** for submission.)
- [ ] Feedback section completed from the real Token Factory run.
