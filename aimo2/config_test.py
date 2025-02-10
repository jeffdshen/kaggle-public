"""Tests for config.py."""

from .config import SYSTEM_PARAMS, get_validation_data


def test_system_params():
    for name, system_params in SYSTEM_PARAMS.items():
        assert system_params.name == name
        request_input = system_params.make_input("What is 1 + 1?")
        assert request_input.conversation[-1]["role"] == "user"


def test_validation_data():
    validation_data = get_validation_data()
    assert len(validation_data.correct_answers) == 100
