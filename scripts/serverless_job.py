"""Rewrite the entrypoint for runs on Nebius Serverless Jobs.

Usage (inside the job image):

    python scripts/serverless_job.py --topic "indian tax law for freelancers" --out /outputs

Reads NEBIUS_API_KEY and TAVILY_API_KEY from the environment. All arguments are
forwarded to the standard CLI, so this is just a stable path to point a job at.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemotron_data_forge.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
