import re
import unicodedata

def split_paragraph(paragraph):
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    sentences = [sentence for sentence in sentences if sentence]
    sentence_count = len(sentences)
    return sentences, sentence_count

def normalize_text(text):
    # Comparison key for duplicates: NFC, lower case, one apostrophe (glottal stop) form, no punctuation
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[’‘ʼ`´]", "'", text)
    text = re.sub(r"[^\w\s']", "", text)
    return re.sub(r"\s+", " ", text).strip()
