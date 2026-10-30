import hashlib
import json

import pytest

from tupi_exp.audit_orthography import (
    MAX_EXAMPLES,
    build_report,
    examples,
    fold,
    portuguese_suspects,
    spelling_groups,
)
from tupi_exp.config import HISTORICAL_PATH, V2_PATH


def row(id_, tupi, portugues, fonte="gemini"):
    return {"id": id_, "input": tupi, "output": portugues, "fonte": fonte}


@pytest.fixture
def small_dataset(tmp_path):
    rows = [
        row("hist_0000", "Irei andando Padre meu:", "irei andando, Padre meu;"),
        row("hist_0001", "abá ka’a rupi", "o homem pela mata"),
        row("hist_0002", "Oîo  çüí", "foi dali"),
        row("hist_0003", "", "vazio"),
        row("sint_0000", "abá ka'a rupi", "o homem pela mata", "sintetico"),
        row("sint_0001", "abá ogûatá", "o homem caminhou", "sintetico"),
        row("sint_0002", "abá ogûatá", "o homem caminhou", "sintetico"),
        row("sint_0003", "abá ogûatá", "o homem andou", "sintetico"),
        row("sint_0004", "kunhã ogûata", "a mulher caminhou", "sintetico"),
    ]
    human = [{"id": "hist_0000", "input": rows[0]["input"], "output": "irei andando, Padre meu;"}]
    dataset = tmp_path / "v2.json"
    historical = tmp_path / "historico.json"
    dataset.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    historical.write_text(json.dumps(human, ensure_ascii=False), encoding="utf-8")
    return str(dataset), str(historical)


def test_categories_are_counted_by_group(small_dataset):
    _, summary, _ = build_report(*small_dataset)
    assert summary["Apóstrofo fora do padrão `'`"]["gemini"] == 1
    assert summary["Espaços duplos"]["gemini"] == 1
    assert summary["Vazios"]["gemini"] == 1
    assert summary["Tupi idêntico à tradução humana"]["gemini"] == 1
    assert summary["Duplicatas exatas"]["sintetico"] == 2
    assert summary["Mesmo Tupi com referências diferentes"]["sintetico"] == 3


def test_portuguese_lookalike_is_flagged():
    rows = [
        row("a", "Irei andando Padre meu:", "irei andando, Padre meu;"),
        row("b", "abá ka'a rupi ogûatá", "o homem caminhou pela mata", "sintetico"),
    ]
    assert [item["id"] for item in portuguese_suspects(rows)] == ["a"]


def test_spelling_variants_ignore_case_but_not_diacritics():
    rows = [row("a", "ogûatá Ogûatá", "x"), row("b", "ogûata oguatá", "y")]
    assert fold("Ogûatá") == "oguata"
    groups = dict(spelling_groups(rows))
    assert set(groups["oguata"]) == {"ogûatá", "ogûata", "oguatá"}


def test_report_is_deterministic_and_limits_the_examples():
    first, summary, _ = build_report(V2_PATH, HISTORICAL_PATH)
    second, _, _ = build_report(V2_PATH, HISTORICAL_PATH)
    assert first == second
    assert summary["Vazios"]["gemini"] == 0
    assert len(examples([row(str(i), "abá", "homem") for i in range(50)])) == MAX_EXAMPLES


def test_audit_does_not_touch_the_dataset():
    def digest():
        with open(V2_PATH, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    before = digest()
    build_report(V2_PATH, HISTORICAL_PATH)
    assert digest() == before
