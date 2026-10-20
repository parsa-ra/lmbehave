import pytest
from lmms_eval.tasks.representation_dynamics_text import utils


def test_arc_numeric_labels_are_mapped_by_choice_position():
    doc = {
        "question": "Which value is even?",
        "choices": {"label": ["1", "2", "3", "4"], "text": ["1", "2", "3", "5"]},
        "answerKey": "2",
    }
    assert utils.arc_doc_to_target(doc) == "B"
    assert utils.arc_process_results(doc, ["B."]) == {"accuracy": 1.0}


def test_choice_parser_does_not_guess_when_response_has_no_answer():
    doc = {"narrative": "Facts.", "question": "Who?", "choices": ["Ada", "Ben"], "answer_index": 0}
    assert utils.musr_process_results(doc, ["I am unsure."]) == {"accuracy": 0.0}


def test_musr_decodes_pinned_string_choice_schema():
    doc = {
        "narrative": "Facts.",
        "question": "Who?",
        "choices": "['Madison', 'Christine']",
        "answer_index": 0,
    }
    prompt = utils.musr_doc_to_text(doc)
    assert "Return only one letter: A, B." in prompt
    assert "A. Madison\nB. Christine" in prompt
    assert utils.musr_process_results(doc, ["A"]) == {"accuracy": 1.0}


def test_musr_rejects_malformed_choices_and_target_index():
    malformed = {"narrative": "N", "question": "Q", "choices": "not a list", "answer_index": 0}
    with pytest.raises(ValueError, match="encoded list"):
        utils.musr_doc_to_text(malformed)

    invalid_target = {
        "narrative": "N",
        "question": "Q",
        "choices": "['one', 'two']",
        "answer_index": 2,
    }
    with pytest.raises(ValueError, match="outside"):
        utils.musr_doc_to_target(invalid_target)


def test_choice_parser_uses_last_valid_standalone_letter():
    doc = {"question": "Q", "options": ["x", "y", "z"], "answer": 2}
    assert utils.mmlu_pro_process_results(doc, ["A was tempting; answer: C."]) == {"accuracy": 1.0}

    letter_doc = {"question": "Q", "options": ["w", "x", "y", "z"], "answer": "D"}
    assert utils.mmlu_pro_process_results(letter_doc, ["D"]) == {"accuracy": 1.0}


def test_gsm8k_numeric_normalization():
    doc = {"question": "Q", "answer": "work\n#### 1,200.00"}
    assert utils.gsm8k_doc_to_target(doc) == "1200"
    assert utils.gsm8k_process_results(doc, ["$1,200"]) == {"accuracy": 1.0}


def test_open_answer_normalization_and_aliases():
    nq_doc = {"question": "Q", "answer": ["The United States", "USA"]}
    trivia_doc = {"question": "Q", "answer": {"value": "William Golding", "aliases": ["Golding"]}}
    assert utils.nq_open_process_results(nq_doc, ["the United States."]) == {"accuracy": 1.0}
    assert utils.triviaqa_process_results(trivia_doc, ["Golding"]) == {"accuracy": 1.0}
