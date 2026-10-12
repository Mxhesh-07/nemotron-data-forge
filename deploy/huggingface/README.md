---
title: Nemotron Data Forge
emoji: 🧪
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
license: mit
---

# 🧪 Nemotron Data Forge

Turn any topic into a **graded, de-duplicated, fine-tuning-ready dataset** using
NVIDIA **Nemotron 3** on **Nebius Token Factory**.

Type a topic → Tavily finds fresh sources → Nemotron 3 Super generates training
examples → Nemotron 3 Ultra grades every example for accuracy, helpfulness and
clarity → low scores are rewritten or dropped and near-duplicates removed →
export `dataset.jsonl`, a quality report and a run manifest.

Set `NEBIUS_API_KEY` (required) and `TAVILY_API_KEY` (for live search) in this
Space's **Settings → Variables and secrets**. Set `DEFAULT_MAX_USD` to cap spend.

Full source, docs and tests:
<https://github.com/Mxhesh-07/nemotron-data-forge>
