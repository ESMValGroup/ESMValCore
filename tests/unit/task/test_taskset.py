import dask
import pytest

from esmvalcore import _task
from esmvalcore.preprocessor import PreprocessingTask


@pytest.mark.parametrize(
    (
        "max_parallel_tasks",
        "available_cpu_cores",
        "n_preproc_tasks",
        "scheduler",
        "expected_workers",
    ),
    [
        (8, 128, 100, "distributed", None),  # not using threaded scheduler
        (8, 128, 0, "threads", None),  # not running preproc tasks
        (8, 128, 100, "threads", 16),
        (4, 20, 4, "threading", 5),  # alternative name for threaded scheduler
        (2, 4, 3, "threads", 2),
        (4, 4, 5, "threads", 1),
        (4, 4, 2, "threads", 2),
    ],
)
def test_taskset_get_dask_config(
    mocker,
    max_parallel_tasks: int,
    available_cpu_cores: int,
    n_preproc_tasks: int,
    scheduler: str,
    expected_workers: int | None,
) -> None:
    mocker.patch.object(
        _task,
        "available_cpu_count",
        return_value=available_cpu_cores,
    )

    tasks = _task.TaskSet(
        {
            PreprocessingTask([], name=f"test{i}")
            for i in range(n_preproc_tasks)
        },
    )

    with dask.config.set({"num_workers": None, "scheduler": scheduler}):
        config = tasks._get_dask_config(max_parallel_tasks=max_parallel_tasks)

    if expected_workers is None:
        assert config == {}
    else:
        assert config == {"num_workers": expected_workers}


def test_taskset_get_dask_config_noop(mocker) -> None:
    tasks = _task.TaskSet()

    with dask.config.set({"num_workers": 4, "scheduler": "threads"}):
        config = tasks._get_dask_config(max_parallel_tasks=2)

    assert config == {}
