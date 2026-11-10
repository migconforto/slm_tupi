import json

import pytest

from tupi_exp.config import V2_PATH
from tupi_exp.lexicon import (
    Entry,
    build_lexicon,
    check_no_leak,
    load_lexicon,
    render_prompt,
    render_text,
    select_entries,
)
from tupi_exp.splits import load_split

INSTRUCTION = "Traduza o seguinte texto em Tupi para Português. Não inclua informações adicionais ou conteúdo irrelevante."


class FakeTokenizer:
    """Stands for the chat template: only the message contents matter here."""

    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "\n<sep>\n".join(message["content"] for message in messages)


def row(id_, tupi, portugues, fonte="gemini"):
    return {"id": id_, "input": tupi, "output": portugues, "fonte": fonte}


def lexicon_of(*pairs):
    return {key: Entry(key, translation, (f"id_{key}",)) for key, translation in pairs}


def test_only_pairs_with_one_or_two_tokens_become_entries():
    lexicon = build_lexicon([row("a", "abá", "homem"), row("b", "ixé kunhã", "Eu sou mulher"), row("c", "abá ogûatá ka'a rupi", "x")])
    assert set(lexicon) == {"abá", "ixé kunhã"}


def test_entries_skip_copies_of_the_tupi_word():
    assert build_lexicon([row("a", "apiar", "apiar"), row("b", "Pavuna", "pavuna.")]) == {}


def test_entry_is_the_normalized_token_with_a_clean_translation():
    lexicon = build_lexicon([row("a", "mba’epe?", "que coisa?"), row("b", "oré suí,", "de nós,")])
    assert lexicon["mba'epe"] == Entry("mba'epe", "que coisa?", ("a",))
    assert lexicon["oré suí"].translation == "de nós"


def test_conflicts_keep_the_translation_with_more_evidence_deterministically():
    rows = [row("a", "paka", "paca"), row("b", "paka?", "paca"), row("c", "paka!", "roedor")]
    lexicon = build_lexicon(rows)
    assert lexicon["paka"] == Entry("paka", "paca", ("a", "b"))
    assert build_lexicon(list(reversed(rows))) == lexicon


def test_matching_is_exact_and_keeps_the_order_of_occurrence():
    lexicon = lexicon_of(("kunhã", "mulher"), ("ka'a", "mata"), ("abá", "homem"))
    entries = select_entries("abá, kunhã ogûatá ka’a rupi", lexicon)
    assert [entry.tupi for entry in entries] == ["abá", "kunhã", "ka'a"]
    assert select_entries("Abá ogûata", lexicon) == []


def test_two_token_entries_win_over_single_tokens():
    lexicon = lexicon_of(("kunhã osó", "a mulher foi"), ("kunhã", "mulher"), ("osó", "foi"))
    assert [entry.tupi for entry in select_entries("kunhã osó ka'ape", lexicon)] == ["kunhã osó"]


def test_entries_do_not_repeat_and_respect_the_limit():
    lexicon = lexicon_of(*[(f"w{i}", str(i)) for i in range(12)])
    text = "w0 w0 " + " ".join(f"w{i}" for i in range(12))
    entries = select_entries(text, lexicon, max_entries=8)
    assert [entry.tupi for entry in entries] == [f"w{i}" for i in range(8)]


def test_leave_one_out_removes_an_entry_backed_only_by_the_example():
    rows = [row("a", "pytun", "noite"), row("b", "pytun ogûatá ka'ape", "de noite anda")]
    lexicon = build_lexicon(rows)
    assert select_entries("pytun", lexicon, example_id="a") == []
    assert [entry.tupi for entry in select_entries("pytun ogûatá ka'ape", lexicon, example_id="b")] == ["pytun"]


def test_leave_one_out_keeps_an_entry_with_other_evidence():
    lexicon = build_lexicon([row("a", "paka", "paca"), row("b", "paka?", "paca")])
    assert [entry.tupi for entry in select_entries("paka", lexicon, example_id="a")] == ["paka"]


def test_leave_one_out_falls_back_to_the_other_tokens_of_the_example():
    lexicon = build_lexicon([row("a", "kunhã osó", "a mulher foi"), row("b", "kunhã", "mulher")])
    assert [entry.tupi for entry in select_entries("kunhã osó", lexicon, example_id="a")] == ["kunhã"]


def test_real_train_prompts_never_show_their_own_answer():
    lexicon = load_lexicon()
    with open(V2_PATH, encoding="utf-8") as f:
        by_id = {item["id"]: item for item in json.load(f)}
    train_ids, _ = load_split()
    for row_id in train_ids:
        entries = select_entries(by_id[row_id]["input"], lexicon, example_id=row_id)
        assert all(entry.evidence != (row_id,) for entry in entries)


def test_user_text_follows_the_spec_format():
    entries = [Entry("abá", "homem", ("a",)), Entry("ka'a", "mata", ("b",))]
    assert render_text(" abá ka'a ", entries) == (
        "Glossário (palavras em Tupi):\n- abá = homem\n- ka'a = mata\n\nTexto em Tupi:\nabá ka'a"
    )


def test_glossary_section_is_omitted_without_matches():
    assert render_text("ogûatá", []) == "Texto em Tupi:\nogûatá"


def test_prompt_keeps_the_current_instruction_as_prefix():
    lexicon = lexicon_of(("abá", "homem"))
    sample = {"Tupi": "abá ogûatá", "Português": "o homem andou"}
    prompt = render_prompt(sample, lexicon, src_lng="Tupi", tgt_lng="Português", mode="train", tokenizer=FakeTokenizer(), eos_rep=0)
    assert prompt["full_prompt"].split("\n<sep>\n") == [
        "Você é um assistente de IA muito útil para traduções.",
        f"{INSTRUCTION}\n\nGlossário (palavras em Tupi):\n- abá = homem\n\nTexto em Tupi:\nabá ogûatá",
        "o homem andou",
    ]


def test_inference_uses_the_same_glossary_without_the_answer():
    lexicon = lexicon_of(("abá", "homem"))
    sample = {"Tupi": "abá ogûatá", "Português": "o homem andou"}
    arguments = dict(src_lng="Tupi", tgt_lng="Português", tokenizer=FakeTokenizer())
    train = render_prompt(sample, lexicon, mode="train", eos_rep=0, **arguments)["full_prompt"]
    inference = render_prompt(sample, lexicon, mode="inference", **arguments)["full_prompt"]
    assert train.startswith(inference)
    assert "o homem andou" not in inference


def test_render_prompt_accepts_a_pandas_row():
    pandas = pytest.importorskip("pandas")
    series = pandas.Series({"Tupi": "abá", "Português": "homem", "id": "x"})
    prompt = render_prompt(series, {}, src_lng="Tupi", tgt_lng="Português", mode="inference", tokenizer=FakeTokenizer())
    assert prompt["full_prompt"].endswith("Texto em Tupi:\nabá")


def test_real_lexicon_has_no_validation_leak():
    with open(V2_PATH, encoding="utf-8") as f:
        by_id = {item["id"]: item for item in json.load(f)}
    _, val_ids = load_split()
    check_no_leak(load_lexicon(), [by_id[row_id] for row_id in val_ids])


def test_leak_check_catches_a_validation_sentence():
    lexicon = build_lexicon([row("a", "abá", "homem")])
    with pytest.raises(AssertionError):
        check_no_leak(lexicon, [row("v", "Abá!", "homem")])
    with pytest.raises(AssertionError):
        check_no_leak(lexicon, [row("a", "outra frase", "x")])


def test_lexicon_is_deterministic():
    assert load_lexicon() == load_lexicon()
