"""Streamlit demo UI for Nemotron Data Forge.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import datetime as _dt
import json
import traceback

import streamlit as st

from nemotron_data_forge import config
from nemotron_data_forge.export import (
    EXPORT_FORMATS,
    dataset_card,
    quality_report,
    records_jsonl,
    split_dataset,
)
from nemotron_data_forge.pipeline import run_pipeline


def _secret(name: str, default: str = "") -> str:
    """Read a key from Streamlit secrets (Cloud/Spaces), falling back to env."""
    try:
        value = st.secrets[name]
    except Exception:  # no secrets.toml, or the key is missing
        return default
    return str(value) if value else default


ENV_LLM_KEY = _secret(config.LLM_KEY_ENV, config.NEBIUS_API_KEY)
ENV_TAVILY_KEY = _secret("TAVILY_API_KEY", config.TAVILY_API_KEY)

st.set_page_config(page_title="Nemotron Data Forge", page_icon="🧪", layout="wide")

st.title("🧪 Nemotron Data Forge")
st.caption(
    "Type a topic → Tavily finds fresh sources → NVIDIA Nemotron 3 Super generates examples "
    "→ Nemotron 3 Ultra grades and filters them → export a clean dataset."
)

# --- Sidebar: keys and settings -------------------------------------------
with st.sidebar:
    st.header("🔑 API keys")
    nebius_key = st.text_input(
        config.LLM_KEY_ENV,
        value=ENV_LLM_KEY,
        type="password",
        help=f"From {config.LLM_PROVIDER_LABEL} — required.",
    )
    tavily_key = st.text_input(
        "TAVILY_API_KEY",
        value=ENV_TAVILY_KEY,
        type="password",
        help="From https://tavily.com — required unless you use demo mode.",
    )
    demo_mode = st.checkbox(
        "Demo mode (use bundled sample sources)",
        value=not bool(ENV_TAVILY_KEY),
        help="Skip Tavily and use three bundled sample pages, so you can test without a Tavily key.",
    )

    st.divider()
    st.header("⚙️ Pipeline")
    examples_per_source = st.slider("Examples per source", 1, 5, config.DEFAULT_EXAMPLES_PER_SOURCE)
    max_sources = st.slider("Max sources", 1, 8, config.DEFAULT_MAX_SOURCES)
    keep_threshold = st.slider(
        "Keep threshold (avg score)", 0.0, 10.0, config.DEFAULT_KEEP_THRESHOLD, step=0.5
    )
    min_axis = st.slider(
        "Minimum axis score", 0.0, 10.0, config.DEFAULT_MIN_AXIS_SCORE, step=0.5
    )
    concurrency = st.slider("Concurrency", 1, 8, config.MAX_CONCURRENCY)
    rewrite_rounds = st.slider(
        "Rewrite-on-reject rounds",
        0,
        3,
        config.DEFAULT_REWRITE_ROUNDS,
        help="Regenerate examples the grader rejects, using its reason as guidance. 0 disables.",
    )
    use_embeddings = st.checkbox("Semantic de-duplication (embeddings)", value=True)
    max_usd = st.number_input(
        "Soft spend cap (USD, 0 = none)",
        min_value=0.0,
        value=float(config.DEFAULT_MAX_USD),
        step=0.01,
        format="%.4f",
        help="Once estimated spend reaches this, new sources stop and optional phases are skipped.",
    )
    export_format = st.selectbox(
        "Export format",
        EXPORT_FORMATS,
        help="Record layout for the downloaded dataset: alpaca, sharegpt, openai or raw.",
    )
    val_ratio = st.slider(
        "Validation split",
        0.0,
        0.5,
        0.0,
        step=0.05,
        help="Fraction of examples held out as a validation set for training.",
    )

    st.divider()
    with st.expander(f"Models used ({config.LLM_PROVIDER_LABEL})"):
        st.code(
            f"generator:  {config.GENERATOR_MODEL}\n"
            f"grader:     {config.GRADER_MODEL}\n"
            f"embeddings: {config.EMBEDDING_MODEL}"
        )

# --- Main inputs -----------------------------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input("Topic", value=config.DEFAULT_TOPIC)
with col2:
    fmt = st.selectbox("Format", config.VALID_FORMATS)

run = st.button("🚀 Generate dataset", type="primary", use_container_width=True)

if run:
    if not nebius_key:
        st.error(f"Please provide a {config.LLM_KEY_ENV} in the sidebar.")
        st.stop()
    if not demo_mode and not tavily_key:
        st.error("Please provide a TAVILY_API_KEY, or enable Demo mode.")
        st.stop()

    progress_bar = st.progress(0.0, text="Starting...")

    def on_progress(pct: float, message: str) -> None:
        progress_bar.progress(min(max(pct / 100.0, 0.0), 1.0), text=message)

    try:
        result = run_pipeline(
            topic=topic,
            fmt=fmt,
            examples_per_source=examples_per_source,
            max_sources=max_sources,
            keep_threshold=keep_threshold,
            min_axis_score=min_axis,
            nebius_api_key=nebius_key,
            tavily_api_key=tavily_key,
            use_sample_sources=demo_mode,
            use_embeddings=use_embeddings,
            max_concurrency=concurrency,
            max_rewrite_rounds=rewrite_rounds,
            max_usd=max_usd,
            on_progress=on_progress,
        )
        st.session_state["result"] = result
    except Exception as exc:  # noqa: BLE001 - show a friendly error in the UI
        progress_bar.empty()
        st.error(f"Pipeline failed: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.stop()

result = st.session_state.get("result")

# --- Results ---------------------------------------------------------------
if result is not None:
    stats = result.stats
    manifest = result.manifest
    train_examples, val_examples = split_dataset(result.examples, val_ratio)
    # Record the export choices so the downloaded manifest matches the files,
    # exactly as the CLI does.
    manifest.config["export_format"] = export_format
    manifest.config["val_ratio"] = val_ratio
    manifest.config["train_examples"] = len(train_examples)
    manifest.config["val_examples"] = len(val_examples)
    st.success(f"Done — kept **{stats.kept}** of {stats.generated} generated examples.")
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Kept", stats.kept)
    c2.metric("Not kept", stats.rejected)
    c3.metric("Duplicates removed", stats.duplicates_removed)
    c4.metric("Recovered by rewrite", stats.recovered)
    c5.metric("Est. cost (USD)", f"${manifest.estimated_cost_usd:.4f}")

    if manifest.budget_hit:
        st.warning(
            f"Budget cap of ${manifest.budget_usd:.4f} reached — generation of new sources "
            "and optional phases (rewrite, embeddings) were skipped."
        )

    if manifest.errors:
        with st.expander(f"⚠️ {len(manifest.errors)} warning(s) during the run"):
            for err in manifest.errors:
                st.write(f"- {err}")
    tab_kept, tab_rejected, tab_sources, tab_manifest = st.tabs(
        ["✅ Kept", "❌ Not kept", "🔗 Sources", "📋 Run manifest"]
    )

    with tab_kept:
        if not result.examples:
            st.info("No examples passed the filters. Try lowering the threshold.")
        for i, ex in enumerate(result.examples, 1):
            with st.expander(f"{i}. {ex.instruction}"):
                st.write(ex.response)
                m = st.columns(3)
                m[0].metric("Accuracy", ex.scores.get("accuracy", "-"))
                m[1].metric("Helpfulness", ex.scores.get("helpfulness", "-"))
                m[2].metric("Clarity", ex.scores.get("clarity", "-"))
                st.caption(f"Source: [{ex.source_title}]({ex.source_url})")

    with tab_rejected:
        if not result.rejected:
            st.info("Nothing was dropped. 🎉")
        for i, ex in enumerate(result.rejected, 1):
            with st.expander(f"{i}. {ex.instruction}  ·  avg {ex.average_score}"):
                st.write(ex.response)
                st.caption(f"Reason: {ex.reason or 'n/a'}")
                st.caption(f"Source: {ex.source_url}")

    with tab_sources:
        for s in result.sources:
            st.markdown(f"- [{s.title}]({s.url}) _(relevance {s.score:.2f})_")

    with tab_manifest:
        st.json(manifest.to_dict())

    # --- Downloads ---
    st.divider()
    st.subheader("⬇️ Export")
    jsonl_text = records_jsonl(train_examples, export_format)
    report = quality_report(result.examples, result.rejected, stats, result.sources, manifest)
    card = dataset_card(stats.topic, stats, manifest)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    d1, d2, d3 = st.columns(3)
    d1.download_button(
        f"⬇️ train-{export_format}.jsonl ({len(train_examples)})",
        data=jsonl_text.encode("utf-8"),
        file_name=f"train-{export_format}-{stamp}.jsonl",
        mime="application/json",
        use_container_width=True,
    )
    d2.download_button(
        "⬇️ quality-report.md",
        data=report.encode("utf-8"),
        file_name=f"quality-report-{stamp}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    d3.download_button(
        "⬇️ manifest.json",
        data=json.dumps(manifest.to_dict(), indent=2).encode("utf-8"),
        file_name=f"manifest-{stamp}.json",
        mime="application/json",
        use_container_width=True,
    )
    if val_examples:
        st.download_button(
            f"⬇️ val-{export_format}.jsonl ({len(val_examples)})",
            data=records_jsonl(val_examples, export_format).encode("utf-8"),
            file_name=f"val-{export_format}-{stamp}.jsonl",
            mime="application/json",
        )
    with st.expander("Preview quality report"):
        st.markdown(report)
    with st.expander("Preview Hugging Face dataset card"):
        st.markdown(card)
