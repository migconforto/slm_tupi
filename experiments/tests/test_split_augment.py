import copy
import glob
import json
import os

import pandas as pd
import pytest

from tupi_exp import ROOT
from tupi_exp.config import BASELINE_VAL_SHA256, V2_PATH, V3_PATH
from tupi_exp.split_augment import build_v3, partition, segments, tupi_key
from tupi_exp.splits import ids_hash, load_split, reproduce_split
from utils.utils_train import pre_process


def row(id_, tupi, portugues, fonte="gemini"):
    return {"id": id_, "input": tupi, "output": portugues, "fonte": fonte}


def toy_rows():
    return [
        row("hist_0", "aa bb cc, dd ee ff", "um dois, tres quatro"),
        row("hist_1", "gg hh", "cinco"),
        row("hist_2", "ii jj", "seis"),
        row("hist_3", "kk ll", "sete"),
        row("hist_4", "mm nn", "oito"),
        row("sint_0", "oo pp qq, rr ss tt", "nove, dez", "sintetico"),
        row("hist_5", "uu vv ww, xx yy zz", "onze, doze"),
    ]


TRAIN = ["hist_0", "hist_1", "hist_2", "hist_3", "hist_4", "sint_0"]
VAL = ["hist_5"]


def test_segments_cut_after_strong_and_medium_punctuation():
    assert segments("a b, c d; e: f. g! h? i") == ["a b,", "c d;", "e:", "f.", "g!", "h?", "i"]
    assert segments("em vez di.sso, quero") == ["em vez di.sso,", "quero"]
    assert segments("sem pontuação") == ["sem pontuação"]


def test_partition_keeps_the_original_text_of_each_side():
    assert partition("aa bb, cc dd", "um, dois") == [("aa bb,", "um,", 0, 0), ("cc dd", "dois", 1, 1)]


def test_partition_needs_the_same_number_of_segments():
    assert partition("aa bb, cc dd", "um dois") == []
    assert partition("aa bb, cc dd", "um, dois, tres") == []


def test_partition_needs_two_tupi_words_in_each_segment():
    assert partition("aa, cc dd", "um, dois") == []


def test_contiguous_windows_are_opt_in_and_skip_the_full_sentence():
    text, target = "aa bb, cc dd, ee ff", "um, dois, tres"
    assert len(partition(text, target)) == 3
    pieces = partition(text, target, contiguous_windows=True)
    assert [(first, last) for _, _, first, last in pieces] == [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)]
    assert pieces[3][:2] == ("aa bb, cc dd,", "um, dois,")


def test_only_long_historical_train_sentences_are_partitioned():
    rows, stats, _ = build_v3(toy_rows(), TRAIN, VAL)
    new = rows[len(toy_rows()):]
    assert {item["id_origem"] for item in new} == {"hist_0"}
    assert stats["longos"] == 1
    assert stats["particionados"] == 1
    assert stats["novos_pares"] == 2


def test_new_rows_keep_the_schema_and_add_the_trace_fields():
    new = build_v3(toy_rows(), TRAIN, VAL)[0][len(toy_rows()):]
    assert new[0] == {
        "id": "hist_0_seg0",
        "input": "aa bb cc,",
        "output": "um dois,",
        "fonte": "gemini",
        "origem": "split_aug",
        "id_origem": "hist_0",
        "segmento_idx": 0,
        "segmento_fim": 0,
    }


def test_v2_rows_stay_first_and_untouched():
    original = toy_rows()
    rows = build_v3(copy.deepcopy(original), TRAIN, VAL)[0]
    assert rows[:len(original)] == original


def test_segment_equal_to_a_validation_sentence_is_dropped():
    rows = toy_rows()
    rows.append(row("hist_6", "DD EE FF", "em validação"))
    result, stats, _ = build_v3(rows, TRAIN, VAL + ["hist_6"])
    assert stats["descartados_validacao"] == 1
    assert "dd ee ff" not in {tupi_key(item["input"]) for item in result[len(rows):]}


def test_segment_equal_to_a_train_pair_is_dropped():
    rows = toy_rows()
    rows.append(row("hist_6", "aa bb cc", "já no treino"))
    _, stats, _ = build_v3(rows, TRAIN + ["hist_6"], VAL)
    assert stats["descartados_treino"] == 1
    assert stats["novos_pares"] == 1


def test_repeated_segments_are_kept_once():
    rows = toy_rows()
    rows.append(row("hist_6", "aa bb cc, gg hh ii", "um dois, outro"))
    _, stats, _ = build_v3(rows, TRAIN + ["hist_6"], VAL)
    assert stats["descartados_entre_novos"] == 1


def test_counts_are_consistent_and_deterministic():
    first = build_v3(toy_rows(), TRAIN, VAL)
    second = build_v3(toy_rows(), TRAIN, VAL)
    assert first[:2] == second[:2]
    stats = first[1]
    dropped = stats["descartados_validacao"] + stats["descartados_treino"] + stats["descartados_entre_novos"]
    assert stats["candidatos"] == stats["novos_pares"] + dropped


def test_saved_split_matches_the_code_and_the_baseline():
    train_ids, val_ids = load_split()
    assert (train_ids, val_ids) == reproduce_split()
    assert len(train_ids) == 1382
    assert len(val_ids) == 593
    assert not set(train_ids) & set(val_ids)
    assert ids_hash(val_ids) == BASELINE_VAL_SHA256


def test_split_matches_the_baseline_run_when_its_logs_exist():
    pattern = os.path.join(ROOT, "notebooks", "logs", "peft_128_Tu_Po", "fit_*_dataset_sintetico_aumentado", "test_dataset.json")
    runs = sorted(glob.glob(pattern))
    if not runs:
        pytest.skip("baseline run not available")
    assert ids_hash(list(pd.read_json(runs[0], lines=True)["id"])) == BASELINE_VAL_SHA256


def test_real_v3_has_no_leakage_and_keeps_v2():
    with open(V2_PATH, encoding="utf-8") as f:
        v2 = json.load(f)
    with open(V3_PATH, encoding="utf-8") as f:
        v3 = json.load(f)
    train_ids, val_ids = load_split()
    assert v3[:len(v2)] == v2

    by_id = {item["id"]: item for item in v3}
    val_keys = {tupi_key(by_id[row_id]["input"]) for row_id in val_ids}
    new = v3[len(v2):]
    assert new
    train_keys = {tupi_key(by_id[row_id]["input"]) for row_id in train_ids}
    assert not val_keys & (train_keys | {tupi_key(item["input"]) for item in new})
    assert {item["origem"] for item in new} == {"split_aug"}
    assert all(item["id_origem"] in set(train_ids) for item in new)


def test_pre_process_drops_nothing_from_v3():
    frame = pd.read_json(V3_PATH)
    assert len(pre_process(frame.copy())) == len(frame)
