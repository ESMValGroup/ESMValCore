from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path

import pytest

import esmvalcore
from esmvalcore._task import BaseTask, TaskSet
from esmvalcore.experimental import CFG, Recipe

esmvaltool_sample_data = pytest.importorskip("esmvaltool_sample_data")

CFG.update(esmvaltool_sample_data.get_rootpaths())


@pytest.fixture
def recipe():
    recipe = Recipe(Path(__file__).with_name('recipe_task_test.yml'))
    return recipe


@pytest.mark.use_sample_data
@pytest.mark.parametrize('max_parallel_tasks', [1, 2, 3, 4, 16, None])
def test_run_tasks(max_parallel_tasks, recipe, tmp_path):
    session = CFG.start_session(str(max_parallel_tasks))
    session['output_dir'] = tmp_path
    session['max_parallel_tasks'] = max_parallel_tasks

    _ = recipe.run()

    for task in recipe._engine.tasks:
        # assert not output_file.exists()
        # assert len(ancestor.output_files) == 1
        # assert ancestor.output_files[0].startswith(output_file)
        # assert str(tmp_path / ancestor.name) in input_files
        print(task.name, task.output_files)
        assert task.output_files


@pytest.fixture
def example_tasks():
    """Example tasks for testing the task runners."""
    tasks = TaskSet()
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

    return tasks


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

    monkeypatch.setattr(BaseTask, '_run', _run)
    monkeypatch.setattr(esmvalcore._task, 'Pool', ThreadPool)

    runner(example_tasks)
    print(order)
    assert len(order) == 12
    assert order == sorted(order)
