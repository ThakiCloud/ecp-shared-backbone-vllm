"""CPU unit tests for the gate front (no vLLM, no GPU). Run: python -m pytest -q tests"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sb_gate_api import gate_from_label_logprobs, label_logprobs  # noqa: E402


def lp(pa, pr, ps):
    z = math.log(pa + pr + ps)
    return [math.log(pa) - z, math.log(pr) - z, math.log(ps) - z]


def test_answer_when_margin_reaches_tau():
    out = gate_from_label_logprobs(lp(.6, .3, .1), tau=0.0)
    assert out["action"] == "ANSWER" and out["raw_action"] == "ANSWER"
    assert out["margin"] == pytest.approx(math.log(2))


def test_high_tau_turns_answer_into_best_non_answer():
    out = gate_from_label_logprobs(lp(.6, .1, .3), tau=1.0)
    assert out["action"] == "ABSTAIN" and out["raw_action"] == "ANSWER"


def test_negative_tau_can_answer_below_argmax():
    out = gate_from_label_logprobs(lp(.4, .5, .1), tau=-0.236)
    assert out["action"] == "ANSWER" and out["raw_action"] == "RETRIEVE_MORE"


def test_probs_sum_to_one():
    out = gate_from_label_logprobs(lp(.2, .5, .3), tau=0.0)
    assert sum(out["probs"].values()) == pytest.approx(1.0)


def test_label_logprobs_reads_token_ids_in_label_order():
    top = {"token_id:30": -0.1, "token_id:10": -2.0, "token_id:20": -3.0}
    assert label_logprobs(top, [10, 20, 30]) == [-2.0, -3.0, -0.1]


def test_missing_label_token_is_an_error():
    with pytest.raises(ValueError, match="did not hit the gate path"):
        label_logprobs({"token_id:10": -1.0}, [10, 20, 30])
