"""Deterministic text-only adapters for representation-dynamics experiments.

The adapters request one concise generation per example so the activation
artifact has one stable sample identity. Their scores are protocol checks, not
standard leaderboard scores, because the prompts and decoding differ from the
source benchmark protocols.
"""

from __future__ import annotations

import ast
import re
import string
from functools import partial
from typing import Any, Iterable

from lmms_eval.tasks.hellaswag.utils import (
    process_docs as process_hellaswag_docs,  # noqa: F401
)

LETTERS = string.ascii_uppercase


def _choice_rows(values: Iterable[Any]) -> list[str]:
    return [f"{LETTERS[index]}. {str(value).strip()}" for index, value in enumerate(values)]


def _choice_prompt(question: str, values: Iterable[Any], *, context: str | None = None) -> str:
    rows = _choice_rows(values)
    allowed = ", ".join(LETTERS[: len(rows)])
    pieces = [f"Answer the following multiple-choice question. Return only one letter: {allowed}."]
    if context:
        pieces.append(str(context).strip())
    pieces.extend((str(question).strip(), *rows, "Answer:"))
    return "\n".join(pieces)


def _extract_choice(response: str, number_of_choices: int) -> str | None:
    allowed = set(LETTERS[:number_of_choices])
    matches = re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", str(response).upper())
    valid = [match for match in matches if match in allowed]
    return valid[-1] if valid else None


def _choice_accuracy(target: str, response: str, number_of_choices: int) -> dict[str, float]:
    return {"accuracy": float(_extract_choice(response, number_of_choices) == target)}


def _musr_choices(doc: dict[str, Any]) -> list[str]:
    """Return MuSR choices from its pinned string-encoded list schema."""
    values = doc["choices"]
    if isinstance(values, str):
        try:
            values = ast.literal_eval(values)
        except (SyntaxError, ValueError) as error:
            raise ValueError("MuSR choices are not a valid encoded list") from error
    if not isinstance(values, (list, tuple)):
        raise TypeError("MuSR choices must decode to a list or tuple")
    choices = [str(value).strip() for value in values]
    if not 2 <= len(choices) <= len(LETTERS) or any(not value for value in choices):
        raise ValueError("MuSR choices must contain 2--26 non-empty values")
    return choices


def arc_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    return _choice_prompt(doc["question"], doc["choices"]["text"])


def arc_doc_to_target(doc: dict[str, Any]) -> str:
    labels = [str(value) for value in doc["choices"]["label"]]
    return LETTERS[labels.index(str(doc["answerKey"]))]


def arc_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    return _choice_accuracy(arc_doc_to_target(doc), results[0] if results else "", len(doc["choices"]["text"]))


def hellaswag_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    return _choice_prompt("Which continuation is most plausible?", doc["choices"], context=doc["query"])


def hellaswag_doc_to_target(doc: dict[str, Any]) -> str:
    return LETTERS[int(doc["gold"])]


def hellaswag_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    return _choice_accuracy(hellaswag_doc_to_target(doc), results[0] if results else "", len(doc["choices"]))


def musr_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    return _choice_prompt(doc["question"], _musr_choices(doc), context=doc["narrative"])


def musr_doc_to_target(doc: dict[str, Any]) -> str:
    choices = _musr_choices(doc)
    answer_index = int(doc["answer_index"])
    if not 0 <= answer_index < len(choices):
        raise ValueError("MuSR answer index is outside the decoded choice list")
    return LETTERS[answer_index]


def musr_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    choices = _musr_choices(doc)
    return _choice_accuracy(musr_doc_to_target(doc), results[0] if results else "", len(choices))


def mmlu_pro_process_docs(dataset, subject: str):
    return dataset.filter(lambda row: row["category"] == subject)


process_mmlu_pro_biology = partial(mmlu_pro_process_docs, subject="biology")
process_mmlu_pro_health = partial(mmlu_pro_process_docs, subject="health")
process_mmlu_pro_history = partial(mmlu_pro_process_docs, subject="history")
process_mmlu_pro_law = partial(mmlu_pro_process_docs, subject="law")
process_mmlu_pro_economics = partial(mmlu_pro_process_docs, subject="economics")
process_mmlu_pro_math = partial(mmlu_pro_process_docs, subject="math")
process_mmlu_pro_physics = partial(mmlu_pro_process_docs, subject="physics")
process_mmlu_pro_computer_science = partial(mmlu_pro_process_docs, subject="computer science")
process_mmlu_pro_engineering = partial(mmlu_pro_process_docs, subject="engineering")
process_mmlu_pro_philosophy = partial(mmlu_pro_process_docs, subject="philosophy")


def mmlu_pro_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    return _choice_prompt(doc["question"], doc["options"])


def mmlu_pro_doc_to_target(doc: dict[str, Any]) -> str:
    answer = doc["answer"]
    if isinstance(answer, str):
        normalized = answer.strip().upper()
        if normalized in LETTERS[: len(doc["options"])]:
            return normalized
    return LETTERS[int(answer)]


def mmlu_pro_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    return _choice_accuracy(mmlu_pro_doc_to_target(doc), results[0] if results else "", len(doc["options"]))


def super_gpqa_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    return _choice_prompt(doc["question"], doc["options"])


def super_gpqa_doc_to_target(doc: dict[str, Any]) -> str:
    return str(doc["answer_letter"]).strip().upper()


def super_gpqa_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    return _choice_accuracy(
        super_gpqa_doc_to_target(doc), results[0] if results else "", len(doc["options"])
    )


def gsm8k_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    return (
        "Solve the following arithmetic word problem. Return only the final numeric answer.\n"
        f"Question: {str(doc['question']).strip()}\nAnswer:"
    )


def _last_number(text: str) -> str | None:
    matches = re.findall(r"-?(?:\d[\d,]*)(?:\.\d+)?", str(text).replace("$", ""))
    if not matches:
        return None
    value = matches[-1].replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return "0" if value in {"-0", ""} else value


def gsm8k_doc_to_target(doc: dict[str, Any]) -> str:
    target = _last_number(str(doc["answer"]).split("####")[-1])
    if target is None:
        raise ValueError("GSM8K document has no numeric target")
    return target


def gsm8k_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    prediction = _last_number(results[0] if results else "")
    return {"accuracy": float(prediction == gsm8k_doc_to_target(doc))}


def _normalize_open_answer(value: str) -> str:
    value = str(value).lower().replace("_", " ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def _open_answer_accuracy(response: str, aliases: Iterable[str]) -> dict[str, float]:
    prediction = _normalize_open_answer(response)
    targets = {_normalize_open_answer(alias) for alias in aliases}
    targets.discard("")
    return {"accuracy": float(bool(prediction) and prediction in targets)}


def nq_open_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    return f"Answer the following factual question with only the shortest correct answer.\nQuestion: {doc['question']}\nAnswer:"


def nq_open_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    return _open_answer_accuracy(results[0] if results else "", doc["answer"])


def triviaqa_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    return f"Answer the following factual question with only the shortest correct answer.\nQuestion: {doc['question']}\nAnswer:"


def triviaqa_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    answer = doc["answer"]
    aliases = list(answer.get("aliases", [])) + [answer.get("value", "")]
    return _open_answer_accuracy(results[0] if results else "", aliases)
