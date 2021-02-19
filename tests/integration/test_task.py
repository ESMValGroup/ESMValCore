import multiprocessing
import os
from functools import partial
from multiprocessing.pool import ThreadPool

import pytest

import esmvalcore
from esmvalcore._task import BaseTask, TaskSet


class MockBaseTask(BaseTask):
    def _run(self, input_files):
        tmp_path = self._tmp_path
        output_file = tmp_path / self.name

        msg = ('running {} in thread {}, using input {}, generating {}'.format(
            self.name, os.getpid(), input_files, output_file))
        print(msg)

        # Check that the output is created just once
        assert not output_file.exists()
        output_file.write_text(msg)
        output_file = str(output_file)

        # Check that ancestor results are provided correctly
        assert len(self.ancestors) == len(input_files)
        for ancestor in self.ancestors:
            assert len(ancestor.output_files) == 1
            assert ancestor.output_files[0].startswith(output_file)
            assert str(tmp_path / ancestor.name) in input_files

        return [output_file]


@pytest.fixture
def example_tasks(tmp_path):
    """Example tasks for testing the task runners."""
    tasks = TaskSet()
    for i in range(3):
        task = MockBaseTask(
            name=f'task{i}',
            ancestors=[
                MockBaseTask(name=f'task{i}-ancestor{j}') for j in range(3)
            ],
        )
        for task0 in task.flatten():
            task0.priority = i
            task0._tmp_path = tmp_path
        tasks.add(task)

    return tasks


@pytest.mark.parametrize('mpmethod', ["spawn", "fork"])
@pytest.mark.parametrize('max_parallel_tasks', [1, 2, 3, 4, 16, None])
def test_run_tasks(monkeypatch, tmp_path, max_parallel_tasks, example_tasks,
                   mpmethod):
    """Check that tasks are run correctly."""
    monkeypatch.setattr(esmvalcore._task, 'Pool',
                        multiprocessing.get_context(mpmethod).Pool)
    example_tasks.run(max_parallel_tasks=max_parallel_tasks)

    for task in example_tasks:
        print(task.name, task.output_files)
        assert task.output_files


@pytest.mark.parametrize('runner', [
    TaskSet._run_sequential,
    partial(TaskSet._run_parallel, max_parallel_tasks=1),
])
def test_runner_uses_priority(monkeypatch, runner, example_tasks):
    """Check that the runner tries to respect task priority."""
    order = []

    def _run(self, input_files):
        print(f'running task {self.name} with priority {self.priority}')
        order.append(self.priority)
        return [f'{self.name}_test.nc']

    monkeypatch.setattr(MockBaseTask, '_run', _run)
    monkeypatch.setattr(esmvalcore._task, 'Pool', ThreadPool)

    runner(example_tasks)
    print(order)
    assert len(order) == 12
    assert order == sorted(order)
