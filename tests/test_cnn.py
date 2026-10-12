import copy

import numpy as np
import pytest
import torch

from src import chart_images as ci
from src import cnn
from src import evaluate as ev
from src import train as tr


def test_model_head_and_device():
    m = cnn.build_model(pretrained=False)
    assert m.fc.out_features == 3
    assert cnn.device().type in ("cpu", "cuda")
    with torch.no_grad():
        assert m(torch.zeros(2, 3, 64, 64)).shape == (2, 3)
