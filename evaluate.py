"""
Evaluation runner for OpenIE systems (see EVALUATION.md). The implementation lives in the
``evaluation`` package; this file is the command-line entry point.

Usage (from the cleaned_up/ directory):
    python evaluate.py --config config/eval_descriptive_subset.toml
    python evaluate.py --config config/eval_descriptive_subset.toml --datasets benchie --systems llm
"""

import sys

from evaluation.cli import main

if __name__ == "__main__":
    sys.exit(main())
