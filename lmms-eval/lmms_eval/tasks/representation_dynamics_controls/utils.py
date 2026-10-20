"""Direct-answer task adapters for representation-dynamics controls.

These adapters retain the source datasets and benchmark scoring logic while
standardizing generated multiple-choice answers.  They are deliberately
separate from the upstream tasks so ordinary lmms-eval behavior is unchanged.
"""

from __future__ import annotations

import random
import re
from typing import Any

import datasets

from lmms_eval.tasks.medqa.utils import medqa_doc_to_choice, medqa_doc_to_target
from lmms_eval.tasks.mmmu import utils as mmmu_utils


def medqa_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    options = doc.get("options")
    if isinstance(options, dict):
        rows = [f"{letter}. {str(options[letter]).strip()}" for letter in "ABCDE" if letter in options]
    elif isinstance(options, list):
        rows = [f"{letter}. {str(value).strip()}" for letter, value in zip("ABCDE", options)]
    else:
        rows = [str(options or "")]
    return (
        "Answer the following medical multiple-choice question. Return only the option letter.\n"
        f"Question: {str(doc.get('question', '')).strip()}\n"
        + "\n".join(rows)
        + "\nAnswer:"
    )


def _extract_letter(response: str, choices: list[str]) -> str | None:
    matches = re.findall(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", response.upper())
    valid = [match for match in matches if match in choices]
    return valid[-1] if valid else None


def medqa_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    prediction = _extract_letter(results[0], medqa_doc_to_choice(doc))
    return {"accuracy": float(prediction == medqa_doc_to_target(doc))}


def _clean_gpqa_text(text: str | None) -> str:
    if text is None:
        return " "
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def gpqa_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Match the upstream seeded option shuffle and expose a bare-letter target."""

    def process(doc):
        choices = [
            _clean_gpqa_text(doc["Incorrect Answer 1"]),
            _clean_gpqa_text(doc["Incorrect Answer 2"]),
            _clean_gpqa_text(doc["Incorrect Answer 3"]),
            _clean_gpqa_text(doc["Correct Answer"]),
        ]
        random.shuffle(choices)
        answer = chr(65 + choices.index(_clean_gpqa_text(doc["Correct Answer"])))
        return {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": answer,
        }

    return dataset.map(process)


def mmmu_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    kwargs = dict(lmms_eval_specific_kwargs or {})
    if kwargs.get("format") == "qwen3_vl":
        kwargs.update(
            pre_prompt="Question: ",
            post_prompt="Return only the option letter.",
            open_ended_prompt="Return only the shortest correct word, phrase, or number.",
        )
    else:
        kwargs.update(
            prompt_type="format",
            multiple_choice_prompt="Return only the option letter.",
            open_ended_prompt="Return only the shortest correct word, phrase, or number.",
        )
    return mmmu_utils.mmmu_doc_to_text(doc, kwargs)


def mmmu_doc_to_messages(doc: dict[str, Any], lmms_eval_specific_kwargs=None):
    prompt = re.sub(r"<image \d+>", "<image>", mmmu_doc_to_text(doc, lmms_eval_specific_kwargs))
    visuals = mmmu_utils.mmmu_doc_to_visual(doc)
    pieces = prompt.split("<image>")
    content = []
    for index, visual in enumerate(visuals):
        if index < len(pieces) - 1 and pieces[index].strip():
            content.append({"type": "text", "text": pieces[index].strip()})
        content.append({"type": "image", "url": visual})
    if pieces and pieces[-1].strip():
        content.append({"type": "text", "text": pieces[-1].strip()})
    return [{"role": "user", "content": content}]


mmmu_doc_to_visual = mmmu_utils.mmmu_doc_to_visual
mmmu_process_results = mmmu_utils.mmmu_process_results
mmmu_aggregate_results = mmmu_utils.mmmu_aggregate_results
