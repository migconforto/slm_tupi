# Builds the two datasets of the pipeline from the raw files in data/:
#   dataset_historico.json            historical sentences with the human translation (gold standard, used as test reference)
#   dataset_sintetico_aumentado.json  SFT training set: historical sentences labelled by the teacher (Gemini)
#                                     + synthetic template pairs
# Historical rows share the same "id" in both files, so the test split can recover the gold reference.

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from utils.utils_nlp import normalize_text

DATA = os.path.join(ROOT, "data")

with open(os.path.join(DATA, "gemini-output_dataset.json"), encoding="utf-8") as f:
    gemini_rows = json.load(f)
with open(os.path.join(DATA, "dataset_augmented.json"), encoding="utf-8") as f:
    augmented_rows = json.load(f)

historico, aumentado = [], []
seen = set()

# Historical: "output" is the human translation, "output_gemini" the teacher label
for row in gemini_rows:
    tupi = (row.get("input") or "").strip()
    gold = (row.get("output") or "").strip()
    teacher = (row.get("output_gemini") or "").strip()
    key = normalize_text(tupi)
    if not tupi or not gold or key in seen:
        continue
    seen.add(key)
    row_id = f"hist_{len(historico):04d}"
    historico.append({"id": row_id, "input": tupi, "output": gold})
    if teacher:
        aumentado.append({"id": row_id, "input": tupi, "output": teacher, "fonte": "gemini"})

# Synthetic: rows of dataset_augmented.json without "instruction" (the ones with it are historical)
n_sintetico = 0
for row in augmented_rows:
    if row.get("instruction"):
        continue
    tupi = row["input"].strip()
    portugues = row["output"].strip()
    key = normalize_text(tupi)
    if not tupi or not portugues or key in seen:
        continue
    seen.add(key)
    aumentado.append({"id": f"sint_{n_sintetico:04d}", "input": tupi, "output": portugues, "fonte": "sintetico"})
    n_sintetico += 1

for name, rows in [("dataset_historico.json", historico), ("dataset_sintetico_aumentado.json", aumentado)]:
    with open(os.path.join(DATA, name), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)

print(f"dataset_historico.json: {len(historico)} pares (tradução humana)")
print(f"dataset_sintetico_aumentado.json: {len(aumentado)} pares "
      f"({len(aumentado) - n_sintetico} Gemini + {n_sintetico} sintéticos)")
