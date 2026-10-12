"""LoRA fine-tune a small causal LM on a generated dataset, then compare.

Designed to run on a free Colab T4 (no API keys required). Heavy libraries
(torch/transformers/peft/datasets) are imported lazily, so this module can be
imported and its ``--check`` path used in CI without a GPU stack installed.

Usage:
    python finetune/train_lora.py \
        --dataset finetune/sample_dataset.jsonl \
        --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --epochs 3 --out finetune_output

    # Validate a dataset and print the plan without training:
    python finetune/train_lora.py --dataset outputs/dataset-*.jsonl --check
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import dataset_utils
from .report import ComparisonRow, save, score_rows

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LEN = 512


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="finetune-lora",
        description="LoRA fine-tune a small model on a Nemotron Data Forge dataset and report "
        "before/after quality.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a dataset.jsonl (the pipeline output) or a sample fixture.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Hugging Face model id.")
    parser.add_argument("--out", default="finetune_output", help="Output directory.")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--eval-ratio", type=float, default=0.25)
    parser.add_argument("--eval-examples", type=int, default=8, help="How many eval prompts to generate.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the dataset and print the split without training (no GPU stack needed).",
    )
    return parser.parse_args(argv)


def resolve_dataset(pattern: str) -> str:
    """Resolve a possibly-globbed dataset path to a single file."""
    if any(ch in pattern for ch in "*?["):
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No dataset matched pattern: {pattern}")
        return matches[-1]
    return pattern


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        path = resolve_dataset(args.dataset)
        examples = dataset_utils.load_jsonl(path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    train, eval_set = dataset_utils.split(examples, eval_ratio=args.eval_ratio, seed=args.seed)

    print(f"Dataset: {path}")
    print(f"  examples: {len(examples)} ({dataset_utils.format_counts(examples)})")
    print(f"  train: {len(train)}  eval: {len(eval_set)}")

    if args.check:
        if not train:
            print("error: no training examples after split", file=sys.stderr)
            return 2
        return 0

    try:
        return _train(args, train, eval_set, path)
    except ImportError as exc:  # pragma: no cover - depends on environment
        print(
            "error: missing fine-tuning dependency "
            f"({exc}).\nInstall them with:\n  pip install 'torch>=2.1' 'transformers>=4.40' "
            "'peft>=0.10' 'datasets>=2.18' 'accelerate>=0.30'",
            file=sys.stderr,
        )
        return 3


def _load_stack():  # pragma: no cover - imports heavy libs
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def _train(args, train: list[dict], eval_set: list[dict], path: str) -> int:  # pragma: no cover
    stack = _load_stack()
    torch = stack["torch"]

    tokenizer = stack["AutoTokenizer"].from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = stack["AutoModelForCausalLM"].from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    lora_config = stack["LoraConfig"](
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = stack["get_peft_model"](model, lora_config)
    model.print_trainable_parameters()

    eos = tokenizer.eos_token or ""
    train_texts = [dataset_utils.build_training_text(ex, eos) for ex in train]

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LEN)

    dataset = stack["Dataset"].from_dict({"text": train_texts}).map(
        tokenize, batched=True, remove_columns=["text"]
    )
    collator = stack["DataCollatorForLanguageModeling"](tokenizer=tokenizer, mlm=False)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    training_args = stack["TrainingArguments"](
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        fp16=torch.cuda.is_available(),
        seed=args.seed,
    )
    trainer = stack["Trainer"](
        model=model, args=training_args, train_dataset=dataset, data_collator=collator
    )
    trainer.train()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter to {adapter_dir}")

    rows = _evaluate(model, tokenizer, eval_set[: max(1, args.eval_examples)], args)
    metrics = score_rows(rows)
    meta: dict[str, Any] = {
        "base_model": args.base_model,
        "dataset_path": path,
        "n_train": len(train),
        "n_eval": len(eval_set),
        "epochs": args.epochs,
        "lora_r": args.lora_r,
    }
    md_path, json_path = save(rows, metrics, out_dir, meta)
    print(f"Wrote report: {md_path}")
    print(f"Wrote data:   {json_path}")
    print("Before:", metrics["before"])
    print("After: ", metrics["after"])
    return 0


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:  # pragma: no cover
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _evaluate(model, tokenizer, examples, args) -> list[ComparisonRow]:  # pragma: no cover
    rows: list[ComparisonRow] = []
    for example in examples:
        prompt = dataset_utils.build_prompt(example["instruction"], example.get("input", ""))
        with model.disable_adapter():
            before = _generate(model, tokenizer, prompt, args.max_new_tokens)
        after = _generate(model, tokenizer, prompt, args.max_new_tokens)
        rows.append(
            ComparisonRow(
                instruction=example["instruction"],
                reference=example["output"],
                before=before,
                after=after,
                fmt=example.get("format", ""),
                source_url=example.get("source_url", ""),
            )
        )
        print(f"  evaluated: {example['instruction'][:60]}...")
    return rows


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
