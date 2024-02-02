import pytest

from yptl.utilities import YPTLDict


def test_create_from_dict():
    config = YPTLDict({"type": "Linear", "args": {"in_features": 2}})
    assert config.type == "Linear"
    assert isinstance(config.args, dict)


def test_add_empty_args_dict():
    config = YPTLDict({"type": "Linear"})
    assert isinstance(config.args, dict)
    assert not bool(config.args)


def test_throw_when_type_key_is_missing():
    with pytest.raises(KeyError):
        YPTLDict({})


def test_throw_when_args_is_not_dict():
    with pytest.raises(TypeError):
        YPTLDict({"type": "Linear", "args": 5})


def test_from_type_with_args():
    config = YPTLDict.from_type_with_args("Linear")
    assert isinstance(config.args, dict)
    config = YPTLDict.from_type_with_args("Linear", {"in_features": 2, "out_features": 5})
    assert config.type == "Linear"
    assert isinstance(config.args, dict)
