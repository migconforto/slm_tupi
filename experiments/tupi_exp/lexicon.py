"""Lexical glossary of the exp02 prompt (G2). Built from the train split only (R2.3, source 2)."""
import collections
import json
from dataclasses import dataclass

from tupi_exp.config import LEXICON_MAX_ENTRIES, LEXICON_MAX_TOKENS, SPLIT_SEED, V2_PATH
from tupi_exp.normalize import search_tokens, tupi_key
from tupi_exp.splits import load_split
from utils.utils_train import create_prompt

GLOSSARY_HEADER = "Glossário (palavras em Tupi):"
TEXT_HEADER = "Texto em Tupi:"


@dataclass(frozen=True)
class Entry:
    tupi: str
    translation: str
    evidence: tuple  # ids of the train examples behind this entry


def build_lexicon(train_rows, *, max_tokens=LEXICON_MAX_TOKENS):
    """Entries from train pairs with 1 to max_tokens Tupi tokens. Keeps the translation with the most evidence."""
    candidates = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in sorted(train_rows, key=lambda item: item["id"]):
        tokens = search_tokens(row["input"])
        if not 1 <= len(tokens) <= max_tokens:
            continue
        if tupi_key(row["input"]) == tupi_key(row["output"]):
            continue  # the teacher only copied the word
        translation = row["output"].strip().rstrip(" ,;:.")
        candidates[" ".join(tokens)][translation].append(row["id"])

    lexicon = {}
    for key, translations in candidates.items():
        translation, evidence = min(translations.items(), key=lambda item: (-len(item[1]), item[0]))
        lexicon[key] = Entry(key, translation, tuple(evidence))
    return lexicon


def load_lexicon(seed=SPLIT_SEED):
    """Glossary of a saved split. Rebuilt on demand, so training and evaluation share the same entries."""
    with open(V2_PATH, encoding="utf-8") as f:
        by_id = {row["id"]: row for row in json.load(f)}
    train_ids, _ = load_split(seed)
    return build_lexicon([by_id[row_id] for row_id in train_ids])


def select_entries(tupi, lexicon, *, example_id=None, max_entries=LEXICON_MAX_ENTRIES):
    """Entries found in a sentence, in order of occurrence, without repeats.

    Leave-one-out (R2.5): an entry whose only evidence is the example itself is skipped, otherwise
    the model would learn to copy the answer from the prompt.
    """
    tokens = search_tokens(tupi)
    selected, position = [], 0
    while position < len(tokens) and len(selected) < max_entries:
        step = 1
        for size in range(LEXICON_MAX_TOKENS, 0, -1):
            entry = lexicon.get(" ".join(tokens[position:position + size]))
            if entry and position + size <= len(tokens) and entry.evidence != (example_id,):
                if entry not in selected:
                    selected.append(entry)
                step = size
                break
        position += step
    return selected


def render_text(tupi, entries):
    """User text that follows the current instruction (R2.2). Without entries, the glossary section is omitted."""
    lines = []
    if entries:
        lines += [GLOSSARY_HEADER] + [f"- {entry.tupi} = {entry.translation}" for entry in entries] + [""]
    return "\n".join(lines + [TEXT_HEADER, tupi.strip()])


def render_prompt(sample, lexicon, *, src_lng, tgt_lng, mode, tokenizer, eos_rep=1, example_id=None,
                  max_entries=LEXICON_MAX_ENTRIES):
    """create_prompt with the glossary before the Tupi text. Same function for train, validation and inference."""
    entries = select_entries(sample[src_lng], lexicon, example_id=example_id, max_entries=max_entries)
    extended = dict(sample)
    extended[src_lng] = render_text(sample[src_lng], entries)
    return create_prompt(extended, src_lng=src_lng, tgt_lng=tgt_lng, eos_rep=eos_rep, mode=mode, tokenizer=tokenizer)


def check_no_leak(lexicon, val_rows):
    """Asserts that no entry comes from, or equals, a validation sentence (P4)."""
    val_ids = {row["id"] for row in val_rows}
    val_keys = {tupi_key(row["input"]) for row in val_rows}
    for entry in lexicon.values():
        assert not val_ids & set(entry.evidence), f"validation example behind {entry.tupi!r}"
        assert tupi_key(entry.tupi) not in val_keys, f"entry equal to a validation sentence: {entry.tupi!r}"
