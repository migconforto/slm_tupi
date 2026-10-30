"""Partition of long historical sentences, on the train split only (G4).

Writes data/dataset_augmentation_v3.json, splits/ and experiments/reports/split_augment.md.
Run from experiments/: python -m tupi_exp.split_augment [--contiguous-windows]
"""
import argparse
import json
import os
import re
import statistics

from tupi_exp import REPORTS_DIR
from tupi_exp.config import BASELINE_VAL_SHA256, SPLIT_SEED, V2_PATH, V3_PATH
from tupi_exp.normalize import search_tokens, tupi_key
from tupi_exp.splits import ids_hash, reproduce_split, save_split

SEGMENT_BOUNDARY = re.compile(r"(?<=[,;:.!?])\s+")
MIN_SEGMENT_WORDS = 2
DEFAULT_PERCENTILE = 75
MAX_REPORT_EXAMPLES = 10


def word_count(text):
    return len(search_tokens(text))


def segments(text):
    return [segment for segment in SEGMENT_BOUNDARY.split(text.strip()) if segment]


def partition(tupi, portuguese, *, contiguous_windows=False):
    """Aligned pieces as (tupi, portuguese, first, last). Empty when the alignment is not safe (R4.3)."""
    tupi_parts, portuguese_parts = segments(tupi), segments(portuguese)
    if len(tupi_parts) < 2 or len(tupi_parts) != len(portuguese_parts):
        return []
    if any(word_count(part) < MIN_SEGMENT_WORDS for part in tupi_parts):
        return []
    pieces = [(t, p, index, index) for index, (t, p) in enumerate(zip(tupi_parts, portuguese_parts))]
    if contiguous_windows:
        count = len(tupi_parts)
        for first in range(count):
            for last in range(first + 1, count):
                if last - first + 1 < count:  # the full window is the original sentence
                    pieces.append((" ".join(tupi_parts[first:last + 1]), " ".join(portuguese_parts[first:last + 1]), first, last))
    return pieces


def long_threshold(train_rows, percentile):
    counts = [word_count(row["input"]) for row in train_rows if row["fonte"] == "gemini"]
    return statistics.quantiles(counts, n=100, method="inclusive")[percentile - 1]


def build_v3(rows, train_ids, val_ids, *, percentile=DEFAULT_PERCENTILE, contiguous_windows=False):
    """Returns (v3 rows, statistics, before/after examples). v2 rows stay untouched and first."""
    by_id = {row["id"]: row for row in rows}
    train_rows = [by_id[row_id] for row_id in sorted(train_ids)]
    val_rows = [by_id[row_id] for row_id in sorted(val_ids)]

    threshold = long_threshold(train_rows, percentile)
    historical = [row for row in train_rows if row["fonte"] == "gemini"]
    long_rows = [row for row in historical if word_count(row["input"]) > threshold]

    val_keys = {tupi_key(row["input"]) for row in val_rows}
    train_keys = {tupi_key(row["input"]) for row in train_rows}

    new_rows, examples = [], []
    stats = {
        "limiar_palavras": threshold,
        "percentil": percentile,
        "historicos_treino": len(historical),
        "longos": len(long_rows),
        "particionados": 0,
        "candidatos": 0,
        "descartados_validacao": 0,
        "descartados_treino": 0,
        "descartados_entre_novos": 0,
        "novos_pares": 0,
    }
    seen = set()
    for row in long_rows:
        pieces = partition(row["input"], row["output"], contiguous_windows=contiguous_windows)
        if pieces:
            stats["particionados"] += 1
        kept = []
        for tupi, portuguese, first, last in pieces:
            stats["candidatos"] += 1
            key = tupi_key(tupi)
            if key in val_keys:
                stats["descartados_validacao"] += 1
            elif key in train_keys:
                stats["descartados_treino"] += 1
            elif key in seen:
                stats["descartados_entre_novos"] += 1
            else:
                seen.add(key)
                suffix = f"{first}" if first == last else f"{first}-{last}"
                kept.append({
                    "id": f"{row['id']}_seg{suffix}",
                    "input": tupi,
                    "output": portuguese,
                    "fonte": row["fonte"],
                    "origem": "split_aug",
                    "id_origem": row["id"],
                    "segmento_idx": first,
                    "segmento_fim": last,
                })
        new_rows += kept
        if kept:
            examples.append((row, kept))

    stats["novos_pares"] = len(new_rows)
    dropped = stats["descartados_validacao"] + stats["descartados_treino"] + stats["descartados_entre_novos"]
    assert stats["candidatos"] == stats["novos_pares"] + dropped, "partition counts do not add up"

    augmented_keys = train_keys | {tupi_key(row["input"]) for row in new_rows}
    assert not val_keys & augmented_keys, "validation sentence inside the augmented train set"
    return rows + new_rows, stats, examples


def render_report(stats, examples, n_train, val_hash, contiguous_windows):
    lines = [
        "# Partição de frases históricas longas (`exp03_split_aug`)",
        "",
        f"Saída: `data/dataset_augmentation_v3.json`. Janelas contíguas: {'ligadas' if contiguous_windows else 'desligadas'}.",
        "",
        f"\"Longa\" significa mais de {stats['limiar_palavras']:.0f} palavras em Tupi: percentil {stats['percentil']} das {stats['historicos_treino']} frases históricas do treino.",
        "",
        "| Medida | Valor |",
        "|---|---:|",
        f"| Frases históricas longas no treino | {stats['longos']} |",
        f"| Frases particionadas | {stats['particionados']} |",
        f"| Segmentos candidatos | {stats['candidatos']} |",
        f"| Descartados: iguais a uma frase de validação | {stats['descartados_validacao']} |",
        f"| Descartados: iguais a um par do treino | {stats['descartados_treino']} |",
        f"| Descartados: repetidos entre os novos pares | {stats['descartados_entre_novos']} |",
        f"| Novos pares | {stats['novos_pares']} |",
        f"| Treino de v2 | {n_train} |",
        f"| Treino aumentado | {n_train + stats['novos_pares']} |",
        "",
        f"SHA-256 dos IDs de validação: `{val_hash}`",
        "",
        "Interseção entre os Tupi normalizados de validação e do treino aumentado: 0 (conferido por asserção).",
        "",
        f"## {min(MAX_REPORT_EXAMPLES, len(examples))} exemplos antes/depois",
        "",
    ]
    for row, kept in examples[:MAX_REPORT_EXAMPLES]:
        lines += [f"**`{row['id']}`:** {row['input']!r} → {row['output']!r}", ""]
        lines += [f"- {piece['input']!r} → {piece['output']!r}" for piece in kept]
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Partition long historical sentences of the train split")
    parser.add_argument("--percentile", type=int, default=DEFAULT_PERCENTILE)
    parser.add_argument("--contiguous-windows", action="store_true")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help="Split seed")
    args = parser.parse_args()

    train_ids, val_ids = reproduce_split(V2_PATH, args.seed)
    val_hash = ids_hash(val_ids)
    if args.seed == SPLIT_SEED:
        assert val_hash == BASELINE_VAL_SHA256, f"validation differs from the baseline: {val_hash}"
    save_split(train_ids, val_ids, args.seed)

    with open(V2_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    v3_rows, stats, examples = build_v3(
        rows, train_ids, val_ids, percentile=args.percentile, contiguous_windows=args.contiguous_windows
    )
    with open(V3_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(v3_rows, f, ensure_ascii=False, indent=4)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = render_report(stats, examples, len(train_ids), val_hash, args.contiguous_windows)
    with open(os.path.join(REPORTS_DIR, "split_augment.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(report)

    for name, value in stats.items():
        print(f"{name}: {value}")
    print(f"validação: {len(val_ids)} ids, hash {val_hash}")


if __name__ == "__main__":
    main()
