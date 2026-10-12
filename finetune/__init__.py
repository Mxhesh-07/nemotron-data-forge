"""Before/after fine-tuning demo for datasets produced by Nemotron Data Forge.

The pure helpers here (:mod:`finetune.dataset_utils`, :mod:`finetune.scoring`,
:mod:`finetune.report`) have no heavy dependencies and are unit-tested offline.
Only :mod:`finetune.train_lora` imports torch/transformers, and it does so
lazily so the rest of the project can be imported without a GPU stack.
"""

from __future__ import annotations

__all__ = ["dataset_utils", "scoring", "report"]
