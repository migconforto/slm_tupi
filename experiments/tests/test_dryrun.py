import pytest

from tupi_exp.config import BASELINE_VAL_SHA256, EXPERIMENTS
from tupi_exp.dryrun import CharEstimator, load_model_config, load_tokenizer, run_experiment


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def results(tokenizer):
    config = load_model_config()
    return {name: run_experiment(name, tokenizer, config) for name in EXPERIMENTS}


def test_all_experiments_share_the_baseline_validation(results):
    assert {result["val_hash"] for result in results.values()} == {BASELINE_VAL_SHA256}
    assert {result["n_val"] for result in results.values()} == {593}


def test_no_experiment_truncates(results):
    assert all(result["truncated"] == 0 for result in results.values())
    assert results["exp02_lexicon"]["max_tokens"] <= EXPERIMENTS["exp02_lexicon"].max_seq_length


def test_train_sizes_and_steps(results):
    assert results["exp01_hparams"]["n_train"] == 1382
    assert results["exp01_hparams"]["steps"] == 516
    assert results["exp02_lexicon"]["n_train"] == 1382
    assert results["exp03_split_aug"]["n_train"] == 1412
    assert results["exp03_split_aug"]["steps"] == 528


def test_only_exp02_uses_the_glossary(results):
    assert results["exp02_lexicon"]["entries_histogram"]
    assert not results["exp01_hparams"]["entries_histogram"]
    assert not results["exp03_split_aug"]["entries_histogram"]


def test_dryrun_works_without_the_tokenizer():
    result = run_experiment("exp01_hparams", CharEstimator(), None)
    assert result["n_val"] == 593
