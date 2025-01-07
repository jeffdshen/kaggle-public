import pytest

from postprocess import convert_texts_to_int, extract_boxed_texts, max_weighted_vote


@pytest.mark.parametrize(
    "text, expected",
    [
        ("\\boxed{12345}", ["12345"]),
        ("\\boxed{\\frac{1}{2}}", ["\\frac{1}{2}"]),
        ("abcdef", []),
        ("\\boxed{12345}\n\\boxed{67890}", ["12345", "67890"]),
    ],
)
def test_extract_boxed_texts(text, expected):
    assert extract_boxed_texts(text) == expected


@pytest.mark.parametrize(
    "input_texts, expected",
    [
        (["1", "2", "3"], [1, 2, 3]),
        ([], []),
        (["abc", "1", "def", "2"], [1, 2]),
        (["1.5", "2", "3.14", "4"], [2, 4]),
        (["-1", "-2", "-3"], [-1, -2, -3]),
        (["\\frac{1}{2}", "1+2", "3"], [3]),
        (["999999", "1000000"], [999999, 1000000]),
    ],
)
def test_convert_texts_to_int(input_texts, expected):
    assert convert_texts_to_int(input_texts) == expected


@pytest.mark.parametrize(
    "answers, weights, expected",
    [
        ([[1], [2], [3], [2]], None, 2),
        ([[3, 2, 1], [2, 1], [1, 3]], None, 1),
        ([[2, 2, 1, 3], [1, 2, 2, 3], [1, 5, 5]], None, 2),
        ([[1], [2], [3]], [1.0, 1.0, 1.1], 3),
        ([[1, 3], [2, 2, 2], [1, 5]], [1.0, 1.2, 1.3], 2),
        ([[1, 2], [], [2, 3]], [1.0, 500.0, 1.0], 2),
        ([[1, 2], [2, 3], [3, 1]], None, 1),
        ([[], [], []], None, 0),
        ([], None, 0),
    ],
)
def test_max_weighted_vote(answers, weights, expected):
    assert max_weighted_vote(answers, weights) == expected
