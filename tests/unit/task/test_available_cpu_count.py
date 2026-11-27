import pytest

from esmvalcore import _task


def test_available_cpu_count_linux(mocker):
    mocker.patch.object(_task, "os")
    _task.os.sched_getaffinity.return_value = {0, 1}
    result = _task.available_cpu_count()
    assert result == 2
    _task.os.sched_getaffinity.assert_called_once_with(0)


@pytest.mark.parametrize(
    ("cpu_count", "expected"),
    [
        (None, 1),
        (2, 2),
    ],
)
def test_available_cpu_count_osx(mocker, cpu_count, expected):
    mocker.patch.object(_task, "os")
    if hasattr(_task.os, "sched_getaffinity"):
        del _task.os.sched_getaffinity
    _task.os.cpu_count.return_value = cpu_count
    result = _task.available_cpu_count()
    assert result == expected
    _task.os.cpu_count.assert_called_once_with()
