#!/usr/bin/env python3
from eval_gqa import condition_name, is_correct, normalize_answer, score_rows


def test_normalize_answer():
    assert normalize_answer(" The Large, Elephant! ") == "large elephant"
    assert normalize_answer("An apple") == "apple"
    assert normalize_answer("  BLUE... car ") == "blue car"


def test_is_correct_multi_answer():
    assert is_correct("Child.", ["kid", "child"])
    assert is_correct("the man", "man")
    assert not is_correct("man", ["kid", "child"])


def test_blank_predictions_are_incorrect():
    assert not is_correct("", ["A", "(A)"])
    assert not is_correct("   ", ["A", "(A)"])
    assert not is_correct("__empty_eos__", ["__empty_eos__"])


def test_condition_name():
    assert condition_name({"corruption": "clean", "severity": 0}) == "Clean"
    assert condition_name({"severity_group": "medium", "severity": 2}) == "Medium"
    assert condition_name({"severity": 3}) == "Heavy"


def test_score_rows_core_metrics():
    rows = [
        {"sample_id": "a", "method": "M1", "condition": "clean", "prediction": "No.", "answer": "no"},
        {"sample_id": "b", "method": "M1", "severity_group": "medium", "prediction": "cat", "answers": ["dog"]},
        {"sample_id": "c", "method": "M1", "severity_group": "heavy", "prediction": "the dog", "answer": "dog"},
    ]
    scored, summary = score_rows(rows, bootstrap_resamples=100, bootstrap_seed=7)
    assert [row["correct"] for row in scored] == [1, 0, 1]
    assert summary["n"] == 3
    assert summary["accuracy"] == 2 / 3
    assert summary["Clean"] == 1.0
    assert summary["Medium"] == 0.0
    assert summary["Heavy"] == 1.0
    assert summary["Avg-C"] == 0.5
    assert summary["Avg-All-C"] == 2 / 3
    assert summary["bootstrap"]["resamples"] == 100


if __name__ == "__main__":
    test_normalize_answer()
    test_is_correct_multi_answer()
    test_condition_name()
    test_score_rows_core_metrics()
