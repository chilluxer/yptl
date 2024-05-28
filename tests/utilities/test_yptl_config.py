from __future__ import annotations

import pytest

from yptl.utilities import YPTLConfig


@pytest.fixture()
def yaml_input():
    return {
        "model": {"type": "blabla", "args": {"a": 3}},
        "trainer": {"max_epochs": 3},
    }


def test_raise_if_required_keyword_is_missing(yaml_input):
    with pytest.raises(KeyError):
        _ = YPTLConfig(yaml_input)
