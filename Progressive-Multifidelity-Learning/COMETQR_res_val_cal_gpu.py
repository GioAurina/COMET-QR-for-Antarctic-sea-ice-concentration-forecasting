"""Entrypoint for the modular multifidelity GPU pipeline."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from COMETQR_workflow import main


if __name__ == "__main__":
    main()
