import multiprocessing
import os
import shutil
from contextlib import contextmanager
from functools import partial
from multiprocessing.pool import ThreadPool

import pytest

import esmvalcore
from esmvalcore._task import (
    BaseTask,
    DiagnosticError,
    DiagnosticTask,
    TaskSet,
    _py2ncl,
    _run_task,
)
from esmvalcore.config._diagnostics import DIAGNOSTICS


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


def get_distributed_client_mock(client):
    """Mock `get_distributed_client` to avoid starting a Dask cluster."""

    @contextmanager
    def get_distributed_client():
        yield client

    return get_distributed_client


@pytest.mark.parametrize(['mpmethod', 'max_parallel_tasks'], [
    ('fork', 1),
    ('fork', 2),
    ('fork', 15),
    ('fork', None),
    ('spawn', 2),
])
def test_run_tasks(monkeypatch, max_parallel_tasks, example_tasks, mpmethod):
    """Check that tasks are run correctly."""
    monkeypatch.setattr(
        esmvalcore._task,
        'get_distributed_client',
        get_distributed_client_mock(None),
    )
    monkeypatch.setattr(esmvalcore._task, 'Pool',
                        multiprocessing.get_context(mpmethod).Pool)
    example_tasks.run(max_parallel_tasks=max_parallel_tasks)

    for task in example_tasks:
        print(task.name, task.output_files)
        assert task.output_files


def test_diag_task_updated_with_address(monkeypatch, mocker, tmp_path):
    """Test that the scheduler address is passed to the diagnostic tasks."""
    # Set up mock Dask distributed client
    client = mocker.Mock()
    monkeypatch.setattr(
        esmvalcore._task,
        'get_distributed_client',
        get_distributed_client_mock(client),
    )

    # Create a task
    mocker.patch.object(DiagnosticTask, '_initialize_cmd')
    task = DiagnosticTask(
        script='test.py',
        settings={'run_dir': tmp_path / 'run'},
        output_dir=tmp_path / 'work',
    )

    # Create a taskset
    mocker.patch.object(TaskSet, '_run_sequential')
    tasks = TaskSet()
    tasks.add(task)
    tasks.run(max_parallel_tasks=1)

    # Check that the scheduler address was added to the
    # diagnostic task settings.
    assert 'scheduler_address' in task.settings
    assert task.settings['scheduler_address'] is client.scheduler.address


@pytest.mark.parametrize('runner', [
    TaskSet._run_sequential,
    partial(
        TaskSet._run_parallel,
        scheduler_address=None,
        max_parallel_tasks=1,
    ),
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


@pytest.mark.parametrize('address', [None, 'localhost:1234'])
def test_run_task(mocker, address):
    # Set up mock Dask distributed client
    mocker.patch.object(esmvalcore._task, 'Client')

    task = mocker.create_autospec(DiagnosticTask, instance=True)
    task.products = mocker.Mock()
    output_files, products = _run_task(task, scheduler_address=address)
    assert output_files == task.run.return_value
    assert products == task.products
    if address is None:
        esmvalcore._task.Client.assert_not_called()
    else:
        esmvalcore._task.Client.assert_called_once_with(address)


def test_py2ncl():
    """Test for _py2ncl func."""
    ncl_text = _py2ncl(None, 'tas')
    assert ncl_text == 'tas = _Missing'
    ncl_text = _py2ncl('cow', 'tas')
    assert ncl_text == 'tas = "cow"'
    ncl_text = _py2ncl([1, 2], 'tas')
    assert ncl_text == 'tas = (/1, 2/)'
    ncl_text = _py2ncl({'cow': 22}, 'tas')
    assert ncl_text == 'tas = True\ntas@cow = 22\n'
    with pytest.raises(ValueError) as ex_err:
        _py2ncl([1, "cow"], 'tas')
    assert 'NCL array cannot be mixed type:' in str(ex_err.value)
    with pytest.raises(ValueError) as ex_err:
        _py2ncl({"a": {"cow": 22}})
    assert 'NCL does not support nested dicts:' in str(ex_err.value)


def _get_single_base_task():
    """Test BaseTask basic attributes."""
    task = BaseTask(
        name='task0',
        ancestors=[BaseTask(name=f'task0-ancestor{j}') for j in range(2)],
    )
    return task


def test_base_task_names():
    task = _get_single_base_task()
    assert task.name == 'task0'
    ancestor_names = [anc.name for anc in task.ancestors]
    assert ancestor_names == ['task0-ancestor0', 'task0-ancestor1']


def test_individual_base_task_attrs():
    task = _get_single_base_task()
    assert task.products == set()
    assert task.output_files is None
    assert task.activity is None
    assert task.priority == 0


def _get_single_diagnostic_task(tmp_path, diag_script, write_diag=True):
    """Assemble a simple DiagnosticTask object."""
    diag_output_dir = tmp_path / 'mydiag'
    diag_run_dir = diag_output_dir / 'run_dir'
    diag_settings = {'run_dir': diag_run_dir, 'profile_diagnostic': False}
    if write_diag:
        with open(diag_script, "w", encoding='utf-8') as fil:
            fil.write("import os\n\nprint(os.getcwd())")

    task = DiagnosticTask(
        name='task0',
        ancestors=[BaseTask(name=f'task0-ancestor{j}') for j in range(2)],
        script=diag_script,
        settings=diag_settings,
        output_dir=diag_output_dir,
    )

    return task


def test_py_diagnostic_task_constructor(tmp_path):
    """Test DiagnosticTask basic attributes."""
    diag_script = tmp_path / 'diag_cow.py'
    task = _get_single_diagnostic_task(tmp_path, diag_script)
    assert task.name == 'task0'
    ancestor_names = [anc.name for anc in task.ancestors]
    assert ancestor_names == ['task0-ancestor0', 'task0-ancestor1']
    assert task.script == diag_script
    assert task.settings == {
        'run_dir': tmp_path / 'mydiag' / 'run_dir',
        'profile_diagnostic': False
    }
    assert task.output_dir == tmp_path / 'mydiag'


def test_diagnostic_diag_script_none(tmp_path):
    """Test case when diagnostic script doesn't exist."""
    diag_script = tmp_path / 'diag_cow.py'
    with pytest.raises(DiagnosticError) as err_msg:
        _get_single_diagnostic_task(tmp_path, diag_script, write_diag=False)
    diagnostics_root = DIAGNOSTICS.scripts
    script_file = os.path.abspath(os.path.join(diagnostics_root, diag_script))
    ept = ("Cannot execute script '{}' "
           "({}): file does not exist.".format(script_file, script_file))
    assert ept == str(err_msg.value)


def _get_diagnostic_tasks(tmp_path, diagnostic_text, extension):
    """Assemble Python diagnostic tasks of DiagnosticTasks."""
    diag = 'diag_cow.' + extension
    diag_script = tmp_path / diag
    diag_output_dir = tmp_path / 'mydiag'
    diag_run_dir = diag_output_dir / 'run_dir'
    diag_plot_dir = diag_output_dir / 'plot_dir'
    diag_work_dir = diag_output_dir / 'work_dir'
    diag_settings = {
        'run_dir': diag_run_dir.as_posix(),
        'plot_dir': diag_plot_dir.as_posix(),
        'work_dir': diag_work_dir.as_posix(),
        'profile_diagnostic': False,
        'exit_on_ncl_warning': False
    }

    with open(diag_script, "w", encoding='utf-8') as fil:
        fil.write(diagnostic_text)

    task = DiagnosticTask(
        name='task0',
        ancestors=None,
        script=diag_script.as_posix(),
        settings=diag_settings,
        output_dir=diag_output_dir.as_posix(),
    )

    return task


# skip if no exec
no_ncl = pytest.mark.skipif(shutil.which('ncl') is None,
                            reason="ncl is not installed")
no_rscript = pytest.mark.skipif(shutil.which('Rscript') is None,
                                reason="Rscript is not installed")

CMD_diag = {
    ('ncl', 'ncl'): _py2ncl({'cow': 22}, 'tas'),
    ('Rscript', 'R'): 'var0 <- "zg"',
    ('python', 'py'): "import os\n\nprint('cow')"
}

CMD_diag_fail = {
    ('ncl', 'ncl'): ("cows on the [river]",
                     "An error occurred during execution of NCL script"),
    ('python', 'py'):
    ("import os\n\nprint(cow)", "diag_cow.py failed with return code 1")
}


@pytest.mark.parametrize('executable,diag_text', CMD_diag.items())
@no_ncl
@no_rscript
def test_diagnostic_run_task(monkeypatch, executable, diag_text, tmp_path):
    """Run DiagnosticTask that will not fail."""

    def _run(self, input_filesi=[]):
        print(f'running task {self.name}')

    task = _get_diagnostic_tasks(tmp_path, diag_text, executable[1])
    monkeypatch.setattr(BaseTask, '_run', _run)
    task.run()


@pytest.mark.parametrize('executable,diag_text', CMD_diag_fail.items())
@no_ncl
def test_diagnostic_run_task_fail(monkeypatch, executable, diag_text,
                                  tmp_path):
    """Run DiagnosticTask that will fail."""

    def _run(self, input_filesi=[]):
        print(f'running task {self.name}')

    task = _get_diagnostic_tasks(tmp_path, diag_text[0], executable[1])
    monkeypatch.setattr(BaseTask, '_run', _run)
    with pytest.raises(DiagnosticError) as err_mssg:
        task.run()
        assert diag_text[1] in str(err_mssg.value)
