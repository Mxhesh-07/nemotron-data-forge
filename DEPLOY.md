# Deploying the live demo

The hackathon requires a **working demo that judges can use for free until
judging ends**. The Streamlit app is ready to host as-is; it reads API keys from
the host's secrets (Streamlit `st.secrets`) and falls back to environment
variables / `.env` for local runs.

Two keys are needed:

- `NEBIUS_API_KEY` — always required (generation, grading, embeddings).
- `TAVILY_API_KEY` — required for live web search; **demo mode** uses three
  bundled sample pages instead, but grading still calls Nemotron.

> Tip: set `DEFAULT_MAX_USD` (or the `max_usd` field in the app) to a small value
> such as `1.0` so a public demo cannot run up a large bill.

---

## Option A — Streamlit Community Cloud (recommended, free)

1. Push this repo to GitHub (already done for `Mxhesh-07/nemotron-data-forge`).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **New app** → pick the repo, branch `main`, main file path `app.py`.
4. Open **Advanced settings → Secrets** and paste:

   ```toml
   NEBIUS_API_KEY = "your-token-factory-key"
   TAVILY_API_KEY = "your-tavily-key"
   # Optional: cap spend for the public demo
   DEFAULT_MAX_USD = "1.0"
   ```

5. Deploy. Streamlit installs `requirements.txt`, then runs `app.py` from the
   repo root, so `import nemotron_data_forge` works without extra steps.

Keep the keys funded until judging ends (Dec 15, 2026) so the demo stays live.

---

## Option B — Hugging Face Spaces (Streamlit SDK)

Spaces use the repo root as the app root and require YAML frontmatter at the top
of `README.md`, which we do **not** want in the GitHub README. So publish the
Space from a small copy:

```bash
# 1. Create an empty Streamlit Space named nemotron-data-forge at https://huggingface.co/new-space
git clone https://huggingface.co/spaces/<your-user>/nemotron-data-forge hf-space
cd hf-space

# 2. Copy the app, package, config and requirements from this repo
cp -r ../nemotron-data-forge/nemotron_data_forge .
cp ../nemotron-data-forge/app.py .
cp ../nemotron-data-forge/requirements.txt .
mkdir -p .streamlit
cp ../nemotron-data-forge/.streamlit/config.toml .streamlit/
cp ../nemotron-data-forge/deploy/huggingface/README.md README.md

# 3. Add the two secrets as Space secrets (Settings → Variables and secrets),
#    then push:
git add -A && git commit -m "Deploy Nemotron Data Forge" && git push
```

The Space template already contains the frontmatter (`sdk: streamlit`,
`app_file: app.py`). Set `NEBIUS_API_KEY`, `TAVILY_API_KEY` and optionally
`DEFAULT_MAX_USD` as secrets.

---

## Local container (any host)

```bash
docker build -t nemotron-data-forge .
docker run --rm -p 8501:8501 \
  -e NEBIUS_API_KEY=... -e TAVILY_API_KEY=... -e DEFAULT_MAX_USD=1.0 \
  --entrypoint streamlit \
  nemotron-data-forge run app.py --server.address 0.0.0.0
```

The image entrypoint runs the headless CLI, so `docker run ... <cli args>`
builds a dataset. Use `--entrypoint streamlit` as shown to serve the web UI
instead (appending `streamlit run ...` to the default entrypoint would forward
those words to the CLI as arguments).

---

## Pre-flight checklist

- [ ] Both keys set as host secrets (not committed).
- [ ] `DEFAULT_MAX_USD` set for the public demo.
- [ ] Demo mode off (so Tavily runs live and the Tavily bonus applies).
- [ ] Open the URL in a fresh browser and run one topic end to end.
- [ ] Keys stay funded through the judging window (ends Dec 15, 2026).
