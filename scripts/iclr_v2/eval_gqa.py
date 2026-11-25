#!/usr/bin/env python3
"""Canonical GQA scorer for ICLR-DUG-V2.

Policy:
- Normalize predictions and answers by lowercasing, punctuation removal, article
  removal (a/an/the), and whitespace collapse.
- Correctness is exact match against any normalized gold answer.
- Report Clean, Medium, Heavy, Avg-C, and Avg-All-C with 5000 bootstrap
  resamples by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import string
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

ARTICLES = {"a", "an", "the"}
PUNCT_TABLE = str.maketrans("", "", string.punctuation)
CONDITION_ORDER = ("Clean", "Medium", "Heavy")
EMPTY_EOS_SENTINELS = {"__empty_eos__"}


def is_empty_prediction(prediction: object) -> bool:
    if prediction is None:
        return True
    text = str(prediction).strip()
    return not text or text.lower() in EMPTY_EOS_SENTINELS


def normalize_answer(text: object) -> str:
    text = str(text).strip().lower()
    text = text.translate(PUNCT_TABLE)
    tokens = [token for token in text.split() if token not in ARTICLES]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def as_gold_list(answers: object) -> list[object]:
    if isinstance(answers, str):
        return [answers]
    if isinstance(answers, Iterable):
        return list(answers)
    return [answers]


NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
}


def normalize_count_answer(text: object) -> str:
    normalized = normalize_answer(text)
    return NUMBER_WORDS.get(normalized, normalized)


def is_correct(prediction: object, answers: object) -> bool:
    if is_empty_prediction(prediction):
        return False
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(gold) for gold in as_gold_list(answers))


def vqa_soft_accuracy(prediction: object, answers: object) -> float:
    if is_empty_prediction(prediction):
        return 0.0
    pred = normalize_answer(prediction)
    matches = sum(1 for gold in as_gold_list(answers) if pred == normalize_answer(gold))
    return min(1.0, matches / 3.0)


def tallyqa_accuracy(prediction: object, answer: object) -> float:
    if is_empty_prediction(prediction):
        return 0.0
    return float(normalize_count_answer(prediction) == normalize_count_answer(answer))


def item_score(item: dict, gold: object) -> float:
    dataset = str(item.get("dataset", "")).strip().lower()
    if dataset in {"textvqa", "vizwiz"}:
        return vqa_soft_accuracy(item["prediction"], gold)
    if dataset == "tallyqa":
        return tallyqa_accuracy(item["prediction"], gold)
    return float(is_correct(item["prediction"], gold))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "sample_id" not in item:
                item["sample_id"] = item.get("question_id", f"row_{line_no:08d}")
            rows.append(item)
    return rows


def condition_name(item: dict) -> str:
    raw = str(item.get("condition", item.get("severity_group", item.get("corruption_level", "")))).strip().lower()
    corruption = str(item.get("corruption_type", item.get("corruption", ""))).strip().lower()
    severity = item.get("severity")
    if raw in {"clean", "none"} or corruption == "clean" or severity == 0:
        return "Clean"
    if raw in {"heavy", "heldout_heavy", "held-out heavy"}:
        return "Heavy"
    if raw in {"medium", "train", "val", "validation"}:
        return "Medium"
    try:
        return "Heavy" if int(severity) >= 3 else "Medium"
    except Exception:
        return "Medium"


def metric_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def bootstrap_ci(values: list[float], resamples: int, seed: int) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    rng = random.Random(seed)
    n = len(values)
    boot = []
    for _ in range(resamples):
        boot.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    boot.sort()
    lo_idx = int(0.025 * (resamples - 1))
    hi_idx = int(0.975 * (resamples - 1))
    return {
        "mean": mean(values),
        "std": pstdev(boot) if len(boot) > 1 else 0.0,
        "ci95_low": boot[lo_idx],
        "ci95_high": boot[hi_idx],
    }


def avg_present(metrics: dict[str, float | None], names: Iterable[str]) -> float | None:
    values = [metrics[name] for name in names if metrics.get(name) is not None]
    return mean(values) if values else None


def score_rows(rows: list[dict], bootstrap_resamples: int = 5000, bootstrap_seed: int = 20260920) -> tuple[list[dict], dict]:
    scored = []
    by_condition: dict[str, list[int]] = defaultdict(list)
    by_corruption: dict[str, list[int]] = defaultdict(list)
    by_method: dict[str, list[int]] = defaultdict(list)
    by_tallyqa_split: dict[str, list[float]] = defaultdict(list)
    for item in rows:
        if "prediction" not in item:
            raise KeyError("prediction")
        if "answer" not in item and "answers" not in item and "gt" not in item:
            raise KeyError("answer/answers/gt")
        gold = item.get("answers", item.get("answer", item.get("gt")))
        correct = item_score(item, gold)
        condition = condition_name(item)
        method = str(item.get("method", "unknown"))
        corruption = "clean" if condition == "Clean" else str(item.get("corruption_type", item.get("corruption", "unknown")))
        row = dict(item)
        row["condition"] = condition
        row["normalized_prediction"] = normalize_answer(item["prediction"])
        row["normalized_answers"] = [normalize_answer(gold_item) for gold_item in as_gold_list(gold)]
        row["correct"] = correct
        scored.append(row)
        by_condition[condition].append(correct)
        by_corruption[corruption].append(correct)
        by_method[method].append(correct)
        if str(item.get("dataset", "")).strip().lower() == "tallyqa":
            split = str(item.get("tallyqa_split") or ("simple" if item.get("issimple") else "complex"))
            by_tallyqa_split[split.capitalize()].append(correct)

    condition_acc = {name: metric_mean(by_condition[name]) for name in CONDITION_ORDER}
    corruption_acc = {key: metric_mean(vals) for key, vals in sorted(by_corruption.items())}
    avg_c = avg_present(condition_acc, ["Medium", "Heavy"])
    avg_all_c = avg_present(condition_acc, CONDITION_ORDER)
    all_correct = [row["correct"] for row in scored]
    summary = {
        "n": len(scored),
        "accuracy": metric_mean(all_correct) or 0.0,
        "Clean": condition_acc["Clean"],
        "Medium": condition_acc["Medium"],
        "Heavy": condition_acc["Heavy"],
        "Avg-C": avg_c,
        "Avg-All-C": avg_all_c,
        "by_condition": {
            name: {"n": len(by_condition[name]), "accuracy": condition_acc[name]} for name in CONDITION_ORDER
        },
        "by_corruption": {key: {"n": len(vals), "accuracy": metric_mean(vals)} for key, vals in sorted(by_corruption.items())},
        "by_method": {key: {"n": len(vals), "accuracy": metric_mean(vals)} for key, vals in sorted(by_method.items())},
        "tallyqa": {
            key: {"n": len(vals), "accuracy": metric_mean(vals)} for key, vals in sorted(by_tallyqa_split.items())
        },
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "overall": bootstrap_ci(all_correct, bootstrap_resamples, bootstrap_seed),
            "by_condition": {
                name: bootstrap_ci(by_condition[name], bootstrap_resamples, bootstrap_seed + idx + 1)
                for idx, name in enumerate(CONDITION_ORDER)
            },
        },
        "normalization": {
            "lowercase": True,
            "strip_punctuation": True,
            "remove_articles": sorted(ARTICLES),
            "collapse_whitespace": True,
            "exact_match": True,
            "empty_prediction_policy": "empty string, whitespace-only string, and __empty_eos__ sentinel are always incorrect",
            "textvqa_soft_accuracy": "min(1, normalized_answer_matches/3) for TextVQA rows",
            "tallyqa_number_words_0_to_20": True,
        },
        "aggregation": {
            "Clean": "micro_accuracy_over_rows_with_clean_condition",
            "Medium": "micro_accuracy_over_rows_with_train_or_validation_corruption_severities",
            "Heavy": "micro_accuracy_over_rows_with_heldout_heavy_corruption_severities",
            "Avg-C": "macro_average_of_available_Medium_and_Heavy",
            "Avg-All-C": "macro_average_of_available_Clean_Medium_Heavy",
            "bootstrap_resamples": bootstrap_resamples,
        },
    }
    return scored, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path, help="JSONL rows with prediction and answer/answers/gt")
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--output-scored", type=Path, help="CSV scored per-sample predictions")
    parser.add_argument("--output-scored-jsonl", type=Path, help="JSONL scored per-sample predictions")
    parser.add_argument("--bootstrap-resamples", default=5000, type=int)
    parser.add_argument("--bootstrap-seed", default=20260920, type=int)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    scored, summary = score_rows(rows, args.bootstrap_resamples, args.bootstrap_seed)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_scored:
        write_csv(args.output_scored, scored)
    if args.output_scored_jsonl:
        write_jsonl(args.output_scored_jsonl, scored)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
