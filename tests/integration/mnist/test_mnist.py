import argparse
from unittest import mock

import pytest

from yptl.scripts.yptl import main


@pytest.mark.usefixtures("_change_cwd_to_test_dir")
@mock.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(template="yptl_mnist_template.yaml"))
def test_yptl_mnist(mock_args):
    main()
