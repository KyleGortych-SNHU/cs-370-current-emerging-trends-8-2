import sys
from unittest.mock import MagicMock

# Mock tensorflow before importing modules
sys.modules["tensorflow"] = MagicMock()

import numpy as np

from GameExperience import GameExperience
from TreasureMaze import TreasureMaze


# Dummy model
class DummyTensor:
    def __init__(self, arr):
        self._arr = arr

    def numpy(self):
        return self._arr


class DummyModel:
    def __init__(self, output_size):
        self.output_shape = (None, output_size)

    def __call__(self, x, training=False):
        batch = x.shape[0]
        return DummyTensor(np.ones((batch, self.output_shape[-1]), dtype=np.float32))


# GameExperience tests
def test_memory_limit():
    model = DummyModel(4)
    exp = GameExperience(model, model, max_memory=2)

    exp.remember([1])
    exp.remember([2])
    exp.remember([3])

    assert len(exp.memory) == 2


def test_predict_shape():
    model = DummyModel(4)
    exp = GameExperience(model, model)

    state = np.zeros(10)
    result = exp.predict(state)

    assert result.shape == (4,)


def test_get_data():
    model = DummyModel(4)
    exp = GameExperience(model, model)

    for _ in range(5):
        exp.remember([
            np.zeros((1, 10)),
            1,
            1.0,
            np.zeros((1, 10)),
            False
        ])

    inputs, targets = exp.get_data(3)

    assert inputs.shape == (3, 10)
    assert targets.shape == (3, 4)


# TreasureMaze tests
def test_maze_runs():
    maze = np.ones((3, 3))
    env = TreasureMaze(maze)

    state, reward, status = env.act(2)

    assert isinstance(state, np.ndarray)
    assert isinstance(reward, float)
    assert status in ["win", "lose", "not_over"]


def test_win_condition():
    maze = np.ones((2, 2))
    env = TreasureMaze(maze)

    env.state = (1, 1, "valid")

    assert env.game_status() == "win"
