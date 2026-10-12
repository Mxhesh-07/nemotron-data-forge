# LoRA before/after — measured results

This records a **real** end-to-end run of the fine-tuning harness so the report
generator and metrics are verified, with honest numbers and honest caveats.

## Environment

| | |
| --- | --- |
| Hardware | CPU only (no GPU), Windows, Python 3.10 |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Adapter | LoRA, `r=8`, `alpha=16`, `dropout=0.05`, `q_proj`/`v_proj` |
| Training | 3 epochs, `grad_accum=8`, `seed=42` |
| Dataset | `outputs/dataset-20261012-231434.jsonl` (the 9-example tax-freelancer run) |
| Split | 6 train / 2 eval (the run's `--val-ratio 0.1` split fed training; eval here is a further 25% hold-out of that file) |
| Command | `python -m finetune.train_lora --dataset outputs/dataset-20261012-231434.jsonl --base-model Qwen/Qwen2.5-0.5B-Instruct --epochs 3 --out finetune_output` |
| Runtime | ~2 min total (23 s of training on CPU) |

## Metrics (mean over the 2 eval examples)

| Metric | Before | After | Delta |
| --- | --- | --- | --- |
| Token F1 vs reference | 0.1905 | 0.2067 | +0.0162 ▲ |
| Keyword recall | 0.7125 | 0.7750 | +0.0625 ▲ |
| Length closeness | 0.0 | 0.0 | 0.0 = |
| Format adherence | 1.0 | 1.0 | 0.0 = |
| Avg words | 93.0 | 93.0 | 0.0 = |

The full side-by-side answers are in `finetune_output/before_after.md` /
`.json` (gitignored; regenerate with the command above).

## Honest interpretation

- This verifies the **harness**: dataset loading, LoRA training, adapter-off vs
  adapter-on generation, offline metrics, and the Markdown/JSON report all work
  end to end and the run exits 0.
- Token-F1 and keyword recall rose, but the deltas are small and the absolute
  quality is low — a 0.5B base model with **6 training examples** and 3 epochs
  cannot learn domain facts. Do **not** read this as a dataset-quality claim.
- The intended before/after demo is `colab_before_after.ipynb` on a **free T4**,
  with a larger instruction model and the full generated dataset. Use this smoke
  test only to confirm the pipeline is wired correctly before that run.

