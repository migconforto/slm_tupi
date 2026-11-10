"""Read-only orthography audit (R3.1). Writes experiments/reports/orthography_audit.md.

Run from experiments/: python -m tupi_exp.audit_orthography
"""
import argparse
import collections
import difflib
import hashlib
import json
import os
import statistics
import string
import unicodedata

from tupi_exp import REPORTS_DIR
from tupi_exp.config import HISTORICAL_PATH, V2_PATH
from tupi_exp.normalize import APOSTROPHE_VARIANTS, search_tokens

GROUPS = (("Históricos", "gemini"), ("Sintéticos", "sintetico"))
SIDES = (("Tupi", "input"), ("Português", "output"))
MAX_EXAMPLES = 20
NEAR_IDENTICAL = 0.8
PORTUGUESE_SHARE = 0.6

APOSTROPHES = ("'",) + tuple(APOSTROPHE_VARIANTS)
HYPHENS = ("-", "‐", "‑", "‒", "–", "—", "−")
HISTORICAL_LETTERS = "äëïöüÄËÏÖÜàèìòùÀÈÌÒÙçÇ"


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def lower_tokens(text):
    return [token.lower() for token in search_tokens(text)]


def fold(text):
    """Comparison key: no case, no diacritics, no apostrophes."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    kept = (char for char in decomposed if not unicodedata.category(char).startswith("M"))
    return "".join(char for char in kept if char not in APOSTROPHES)


def show_char(char):
    if char == "|":
        return "`\\|`"
    if char == "`":
        return "`` ` ``"
    if char.isspace():
        return "(espaço)" if char == " " else "(branco)"
    if unicodedata.category(char).startswith("M"):
        return f"`◌{char}`"
    return f"`{char}`"


def pair(row):
    return f"`{row['id']}`: {row['input']!r} → {row['output']!r}"


def examples(rows, describe=pair):
    return [f"- {describe(row)}" for row in rows[:MAX_EXAMPLES]]


def inventory_table(texts):
    counts = collections.Counter(char for text in texts for char in text)
    plain = string.ascii_letters + string.digits
    total = sum(count for char, count in counts.items() if char in plain)
    lines = [f"Letras e dígitos ASCII: {total} ocorrências.", "", "| Caractere | Codepoint | Nome | Frequência |", "|---|---|---|---:|"]
    others = [(char, count) for char, count in counts.items() if char not in plain]
    for char, count in sorted(others, key=lambda item: (-item[1], item[0])):
        lines.append(f"| {show_char(char)} | U+{ord(char):04X} | {unicodedata.name(char, '?')} | {count} |")
    return lines


def inventory_section(groups):
    lines = ["## 2. Inventário de caracteres", ""]
    for title, group in groups:
        for side, field in SIDES:
            lines += [f"### {title}: lado {side}", ""] + inventory_table([row[field] for row in group]) + [""]
    return lines


def unicode_section(groups, tally):
    lines = ["## 3. Formas Unicode", "", "| Grupo | Lado | Textos fora de NFC | Caracteres combinantes |", "|---|---|---:|---:|"]
    details = []
    for title, group in groups:
        for side, field in SIDES:
            not_nfc = [row for row in group if unicodedata.normalize("NFC", row[field]) != row[field]]
            combining = collections.Counter(
                char for row in group for char in row[field] if unicodedata.category(char).startswith("M")
            )
            lines.append(f"| {title} | {side} | {len(not_nfc)} | {sum(combining.values())} |")
            tally(f"Fora de NFC ({side})", not_nfc)
            details += [f"- {title}, {side}: {show_char(char)} U+{ord(char):04X}, {n} vezes" for char, n in combining.most_common(MAX_EXAMPLES)]
            details += examples(not_nfc)
    return lines + ["", "Combinantes presentes e exemplos fora de NFC:", ""] + (details or ["- Nenhum."]) + [""]


def marks_section(groups, tally):
    lines = ["## 4. Apóstrofo (oclusiva glotal) e hífens", "", "Lado Tupi.", "", "| Grupo | Sinal | Codepoint | Ocorrências | Linhas |", "|---|---|---|---:|---:|"]
    for title, group in groups:
        for mark in APOSTROPHES + HYPHENS:
            hits = [row for row in group if mark in row["input"]]
            if hits:
                total = sum(row["input"].count(mark) for row in hits)
                lines.append(f"| {title} | {show_char(mark)} | U+{ord(mark):04X} | {total} | {len(hits)} |")
    lines.append("")
    for title, group in groups:
        variants = [row for row in group if any(mark in row["input"] for mark in APOSTROPHE_VARIANTS)]
        mixed = [row for row in group if sum(mark in row["input"] for mark in APOSTROPHES) > 1]
        tally("Apóstrofo fora do padrão `'`", variants)
        tally("Mais de um tipo de apóstrofo na linha", mixed)
        lines += [f"{title}: {len(variants)} linhas com apóstrofo fora do padrão; {len(mixed)} misturam tipos.", ""]
        lines += examples(variants) + [""]
    return lines


def final_mark(text):
    text = text.strip()
    return text[-1] if text and text[-1] in ".,;:!?" else "(nenhuma)"


def layout_section(groups, tally):
    lines = ["## 5. Espaços, pontuação final e maiúsculas", "", "Lado Tupi.", ""]
    for title, group in groups:
        double = tally("Espaços duplos", [row for row in group if "  " in row["input"]])
        edges = tally("Espaços nas pontas", [row for row in group if row["input"] != row["input"].strip()])
        blanks = tally("Outros brancos", [row for row in group if any(char.isspace() and char != " " for char in row["input"])])
        inner = tally("Maiúscula depois da primeira palavra", [row for row in group if any(word[:1].isupper() for word in row["input"].split()[1:])])
        upper_start = sum(row["input"][:1].isupper() for row in group)
        finals = collections.Counter(final_mark(row["input"]) for row in group)
        lines += [
            f"### {title}",
            "",
            f"- Espaços duplos: {len(double)}; nas pontas: {len(edges)}; outros brancos: {len(blanks)}.",
            f"- Começam com maiúscula: {upper_start} de {len(group)}.",
            f"- Maiúscula depois da primeira palavra (possíveis nomes próprios): {len(inner)}.",
            "- Pontuação final: " + ", ".join(f"`{mark}` {n}" for mark, n in finals.most_common()) + ".",
            "",
        ]
        lines += examples(inner) + [""]
    return lines


def length_section(groups):
    lines = ["## 6. Comprimento em palavras", "", "| Grupo | Lado | Mín | P25 | Mediana | P75 | P90 | Máx | Média |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for title, group in groups:
        for side, field in SIDES:
            values = [len(search_tokens(row[field])) for row in group]
            cuts = statistics.quantiles(values, n=100, method="inclusive")
            lines.append(
                f"| {title} | {side} | {min(values)} | {cuts[24]:.0f} | {cuts[49]:.0f} | {cuts[74]:.0f} "
                f"| {cuts[89]:.0f} | {max(values)} | {statistics.mean(values):.1f} |"
            )
    return lines + [""]


def portuguese_suspects(rows):
    """Tupi-side sentences whose words mostly appear only in Portuguese targets."""
    portuguese = {token for row in rows for token in lower_tokens(row["output"]) if len(token) > 1}
    tupi = {token for row in rows if row["fonte"] == "sintetico" for token in lower_tokens(row["input"])}
    suspects = []
    for row in rows:
        tokens = lower_tokens(row["input"])
        hits = [token for token in tokens if token in portuguese and token not in tupi]
        if len(tokens) >= 3 and len(hits) / len(tokens) >= PORTUGUESE_SHARE:
            suspects.append(row)
    return suspects


def near_reference(rows, reference_of):
    """Rows whose Tupi side is identical or almost identical to a reference."""
    identical, near = [], []
    for row in rows:
        reference = reference_of(row)
        if reference is None:
            continue
        source, target = " ".join(lower_tokens(row["input"])), " ".join(lower_tokens(reference))
        if source == target:
            identical.append(row)
        elif difflib.SequenceMatcher(None, source, target).ratio() >= NEAR_IDENTICAL:
            near.append(row)
    return identical, near


def suspicious_section(rows, human_by_id, tally):
    def human_pair(row):
        return f"`{row['id']}`: {row['input']!r} → humana: {human_by_id[row['id']]!r}"

    historical = [row for row in rows if row["fonte"] == "gemini"]
    same_output, near_output = near_reference(rows, lambda row: row["output"])
    same_human, near_human = near_reference(historical, lambda row: human_by_id.get(row["id"]))

    pairs = collections.Counter((row["input"], row["output"]) for row in rows)
    targets = collections.defaultdict(set)
    for row in rows:
        targets[row["input"]].add(row["output"])

    categories = [
        ("Tupi idêntico ao `output`", same_output, pair),
        (f"Tupi quase idêntico ao `output` (similaridade ≥ {NEAR_IDENTICAL})", near_output, pair),
        ("Tupi idêntico à tradução humana", same_human, human_pair),
        (f"Tupi quase idêntico à tradução humana (similaridade ≥ {NEAR_IDENTICAL})", near_human, human_pair),
        ("Tupi que parece Português", portuguese_suspects(rows), pair),
        ("Vazios", [row for row in rows if not row["input"].strip() or not row["output"].strip()], pair),
        ("Duplicatas exatas", [row for row in rows if pairs[(row["input"], row["output"])] > 1], pair),
        ("Mesmo Tupi com referências diferentes", [row for row in rows if len(targets[row["input"]]) > 1], pair),
        (f"Letras de grafia histórica ({HISTORICAL_LETTERS})", historical_spelling(rows), pair),
    ]
    lines = ["## 7. Linhas suspeitas", "", "Nada foi corrigido: as linhas abaixo são só para decisão (R3.4).", ""]
    for name, found, describe in categories:
        tally(name, found)
        lines += [f"### {name}: {len(found)}", ""] + (examples(found, describe) or ["- Nenhuma."]) + [""]
    return lines


def historical_spelling(rows):
    return [row for row in rows if any(char in row["input"] for char in HISTORICAL_LETTERS)]


def spelling_groups(rows):
    """Probable variants of a word: same form without diacritics or apostrophes, more than one spelling."""
    groups = collections.defaultdict(collections.Counter)
    for row in rows:
        for token in lower_tokens(row["input"]):
            groups[fold(token)][token] += 1
    variants = {key: spellings for key, spellings in groups.items() if key and len(spellings) > 1}
    return sorted(variants.items(), key=lambda item: (-sum(item[1].values()), item[0]))


def spelling_section(variants):
    lines = [
        "## 8. Variantes ortográficas prováveis",
        "",
        f"{len(variants)} grupos com mais de uma grafia, sem contar diferenças de caixa. Os {min(MAX_EXAMPLES, len(variants))} mais frequentes:",
        "",
    ]
    for key, spellings in variants[:MAX_EXAMPLES]:
        lines.append(f"- `{key}`: " + ", ".join(f"`{form}` ({n})" for form, n in spellings.most_common()))
    return lines + [""]


def proposals_section(rows):
    spelled = historical_spelling(rows)
    hyphens = [row for row in rows if "-" in row["input"]]
    lines = [
        "## 9. Normalizações",
        "",
        "**Aplicadas** em `normalize_tupi`, só para busca e comparação (o texto de treino não é reescrito): NFC, unificação dos apóstrofos para `'`, colapso de espaços e `strip`.",
        "",
        "**Propostas, não aplicadas** (exigem decisão ou revisão de especialista):",
        "",
    ]
    if spelled:
        lines.append(f"1. Mapear as letras de grafia histórica para a grafia de Navarro ({len(spelled)} linhas). Exemplo: {spelled[0]['input']!r}.")
    if hyphens:
        lines.append(f"2. Unificar o uso de hífen interno ({len(hyphens)} linhas). Exemplo: {hyphens[0]['input']!r}.")
    lines.append("3. Unir as variantes da seção 8 em uma única grafia. Algumas podem ser palavras distintas, por isso a lista pede revisão.")
    lines.append("4. Remover a pontuação final do lado Tupi.")
    return lines + [""]


def build_report(dataset_path, historical_path):
    rows = load_rows(dataset_path)
    human_by_id = {row["id"]: row["output"] for row in load_rows(historical_path)}
    groups = [(title, [row for row in rows if row["fonte"] == fonte]) for title, fonte in GROUPS]
    summary = {}

    def tally(name, found):
        summary.setdefault(name, collections.Counter()).update(row["fonte"] for row in found)
        return found

    body = (
        inventory_section(groups)
        + unicode_section(groups, tally)
        + marks_section(groups, tally)
        + layout_section(groups, tally)
        + length_section(groups)
        + suspicious_section(rows, human_by_id, tally)
    )
    variants = spelling_groups(rows)
    body += spelling_section(variants) + proposals_section(rows)

    with open(dataset_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    header = [
        "# Auditoria de ortografia",
        "",
        f"Arquivo auditado (somente leitura): `{os.path.basename(dataset_path)}`",
        "",
        f"SHA-256: `{digest}`",
        "",
        f"Linhas: {len(rows)} ({len(groups[0][1])} históricas, {len(groups[1][1])} sintéticas). No máximo {MAX_EXAMPLES} exemplos por categoria.",
        "",
        "Nas históricas, o `output` é a tradução do Gemini; a tradução humana vem de `data/dataset_historico.json`.",
        "",
        "## 1. Resumo",
        "",
        "| Categoria | Históricos | Sintéticos |",
        "|---|---:|---:|",
    ]
    table = [f"| {name} | {counts['gemini']} | {counts['sintetico']} |" for name, counts in summary.items()]
    table.append(f"| Grupos de grafias variantes (conjunto todo) | {len(variants)} | |")
    report = "\n".join(header + table + [""] + body) + "\n"
    return report, summary, variants


def main():
    parser = argparse.ArgumentParser(description="Read-only orthography audit of the Tupi side")
    parser.add_argument("--dataset", default=V2_PATH)
    parser.add_argument("--historical", default=HISTORICAL_PATH)
    parser.add_argument("--output", default=os.path.join(REPORTS_DIR, "orthography_audit.md"))
    args = parser.parse_args()

    report, summary, variants = build_report(args.dataset, args.historical)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        f.write(report)
    for name, counts in summary.items():
        print(f"{name}: históricos {counts['gemini']}, sintéticos {counts['sintetico']}")
    print(f"Grupos de grafias variantes: {len(variants)}")
    print(f"Relatório: {args.output}")


if __name__ == "__main__":
    main()
