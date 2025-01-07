from collections import Counter, defaultdict
import re


def extract_boxed_texts(text: str) -> list[str]:
    pattern = r"\\boxed{(.*)}"
    matches = re.findall(pattern, text)
    assert all(isinstance(m, str) for m in matches)
    return matches


def convert_texts_to_int(matches: list[str]) -> list[int]:
    ans: list[int] = []
    for m in matches:
        assert isinstance(m, str)
        try:
            num = int(m)
        except ValueError:
            continue
        ans.append(num)
    return ans


def max_weighted_vote(
    answers: list[list[int]], weights: list[float] | None = None, default: int = 0
) -> int:
    """Calculate the maximum weighted vote.

    Each voter contributes to each of their answers, with the given total weight.
    If a voter did not generate an answer, they contribute no answer.
    In case of ties, return the first most voted answer.
    If no voter generated an answer, return the default value.
    """
    if weights is None:
        weights = [1.0] * len(answers)
    assert len(answers) == len(weights)
    votes: defaultdict[int, float] = defaultdict[int, float](float)
    for answer, weight in zip(answers, weights):
        # Slightly more numerically stable to use Counter
        answer_counts = Counter(answer)
        for num, count in answer_counts.items():
            votes[num] += count / len(answer) * weight

    return max(votes, key=votes.__getitem__, default=default)
