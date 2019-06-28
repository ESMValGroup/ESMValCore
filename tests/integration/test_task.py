from functools import partial
from multiprocessing.pool import ThreadPool

import pytest

import esmvalcore
from esmvalcore._task import (BaseTask, _run_tasks_parallel,
                              _run_tasks_sequential)


@pytest.mark.parametrize('run_tasks', [
    _run_tasks_sequential,
    partial(_run_tasks_parallel, max_parallel_tasks=1),
])
def test_tasks_run_(monkeypatch, run_tasks):

    order = []

    def _run(self, input_files):
        print(f'running task {self.name} with priority {self.priority}')
        order.append(self.priority)
        return [f'{self.name}_test.nc']

    monkeypatch.setattr(BaseTask, '_run', _run)
    monkeypatch.setattr(esmvalcore._task, 'Pool', ThreadPool)

    tasks = set()
    for i in range(3):
        task = BaseTask(
            name=f'task{i}',
            ancestors=[
                BaseTask(name=f'task{i}-ancestor{j}') for j in range(3)
            ],
        )
        for task0 in task.flatten():
            task0.priority = i
        tasks.add(task)

    run_tasks(tasks)
    print(order)
    assert len(order) == 12
    assert order == sorted(order)
