import sys

import pytest

from esmvalcore.exceptions import SuppressedError


@pytest.mark.parametrize('exception', [SuppressedError, ValueError])
def test_suppressedhook(capsys, exception):
    try:
        raise exception('error')
    except exception:
        args = sys.exc_info()
    sys.excepthook(*args)
    msg = capsys.readouterr().err
    if issubclass(exception, SuppressedError):
        assert msg == "SuppressedError: error\n"
    else:
        ending = "ValueError: error\n"
        assert msg.endswith(ending)
        # because `msg` also contains the traceback, it should be
        # longer than `ending`
        assert len(msg) > len(ending)
