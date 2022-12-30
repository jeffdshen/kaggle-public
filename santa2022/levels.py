from typing import Any
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Level:
    colors: Any
    seen: Any
    arm: tuple
    start: tuple
    target: tuple
    remaining: int


def normalize_level(level):
    level.colors = np.array(level.colors, dtype=np.float32)
    level.seen = np.array(level.seen, dtype=np.uint8)
    level.seen[tuple(level.start)] = 1
    level.remaining = np.count_nonzero(level.seen == 0)


def make_level(*args, **kwargs):
    level = Level(*args, **kwargs)
    normalize_level(level)
    return level


def dump_level(path, level):
    with open(path, "w") as f:
        level_dict = asdict(level)
        level_dict["colors"] = level_dict["colors"].tolist()
        level_dict["seen"] = level_dict["seen"].tolist()
        json.dump(level_dict, f)


def import_level(path):
    with open(path) as f:
        level_dict = json.load(f)
        return make_level(**level_dict)


def get_default_levels():
    levels_path = Path(__file__) / ".." / "default_levels"
    levels_path = levels_path.resolve()
    try:
        paths = levels_path.glob("*.json")
        levels = {int(path.stem): import_level(path) for path in paths}
        levels = [levels[i] for i in range(len(levels))]
        if not levels:
            raise RuntimeError("No levels found")
        return levels
    except:
        warnings.warn("Couldn't load levels, defaulting to blank level")
        zero_level = make_level(
            colors=np.ones((64, 64, 3), dtype=np.float32),
            seen=np.zeros((64, 64), dtype=np.uint8),
            arm=(320, 160, 80, 40, 20, 10, 5, 5),
            start=(0, 0),
            target=(63, 63),
            remaining=0,
        )
        return [zero_level]


LEVELS = get_default_levels()
