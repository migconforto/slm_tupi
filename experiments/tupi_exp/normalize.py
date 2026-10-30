"""Safe normalization of the Tupi side (R3.2). Used for lookups and comparisons, never to rewrite the training text.

It complements the metrics normalization, which replaces apostrophes and hyphens with spaces and lowers the case.
"""
import unicodedata

APOSTROPHE_VARIANTS = "’‘ʼ`´"
_APOSTROPHE_TABLE = str.maketrans({variant: "'" for variant in APOSTROPHE_VARIANTS})

# Stripped from the edges of a search token. The apostrophe stays: it is the glottal stop
EDGE_PUNCTUATION = ".,;:!?()[]{}\"“”«»…"


def normalize_tupi(text, *, safe=True):
    """NFC, one apostrophe form, collapsed spaces. Case, diacritics and inner punctuation stay as they are."""
    if not safe:
        raise NotImplementedError("Aggressive rules are only proposed in the audit report, never applied")
    text = unicodedata.normalize("NFC", text).translate(_APOSTROPHE_TABLE)
    return " ".join(text.split())


def search_tokens(text):
    """Words of a sentence, as used for lexicon lookups and word counts."""
    tokens = (token.strip(EDGE_PUNCTUATION) for token in normalize_tupi(text).split())
    return [token for token in tokens if token]
