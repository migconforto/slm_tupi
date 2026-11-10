"""SFT experiments of SPEC_experimentos_sft.md. Additive: nothing here changes the baseline pipeline."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data")
SPLITS_DIR = os.path.join(ROOT, "splits")
REPORTS_DIR = os.path.join(ROOT, "experiments", "reports")

# utils/ lives at the project root
if ROOT not in sys.path:
    sys.path.append(ROOT)
