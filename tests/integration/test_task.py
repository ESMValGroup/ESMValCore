import os
import shutil
from functools import partial
from multiprocessing.pool import ThreadPool

import pytest

import esmvalcore
from esmvalcore._config import DIAGNOSTICS_PATH
from esmvalcore._task import (BaseTask, DiagnosticError, DiagnosticTask,
                              _py2ncl, _run_tasks_parallel,
                              _run_tasks_sequential, run_tasks, which)


@pytest.fixture
def example_tasks():
    """Example tasks for testing the task runners."""
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

    return tasks


@pytest.mark.parametrize('max_parallel_tasks', [1, 2, 3, 4, 16, None])
def test_run_tasks(monkeypatch, tmp_path, max_parallel_tasks, example_tasks):
    """Check that tasks are run correctly."""
    def _run(self, input_files):
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

    monkeypatch.setattr(BaseTask, '_run', _run)

    run_tasks(example_tasks, max_parallel_tasks)

    for task in example_tasks:
        print(task.name, task.output_files)
        assert task.output_files


@pytest.mark.parametrize('runner', [
    _run_tasks_sequential,
    partial(_run_tasks_parallel, max_parallel_tasks=1),
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


@pytest.mark.parametrize('executables', ['ls', 'mv'])
def test_which(executables):
    """Test the which wrapper."""
    assert which(executables).split(os.sep)[-1] in ['ls', 'mv']


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
        _py2ncl({"cow": 22}, None)
    assert 'NCL does not support nested dicts:' in str(ex_err.value)


def _get_single_base_task():
    """Test BaseTask basic attributes."""
    task = BaseTask(
        name='task0',
        ancestors=[BaseTask(name=f'task0-ancestor{j}') for j in range(2)],
    )
    for task0 in task.flatten():
        task0.priority = 0
    tasks = {task}

    return tasks


def test_base_task_names():
    tasks = _get_single_base_task()
    task_names = [task.name for task in tasks]
    assert task_names == ['task0']
    ancestor_names = [anc.name for anc in list(tasks)[0].ancestors]
    assert ancestor_names == ['task0-ancestor0', 'task0-ancestor1']


def test_individual_base_task_attrs():
    tasks = _get_single_base_task()
    task = list(tasks)[0]
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
        with open(diag_script, "w") as fil:
            fil.write("import os\n\nprint(os.getcwd())")

    task = DiagnosticTask(
        name='task0',
        ancestors=[BaseTask(name=f'task0-ancestor{j}') for j in range(2)],
        script=diag_script,
        settings=diag_settings,
        output_dir=diag_output_dir,
    )
    tasks = {task}

    return tasks


def test_py_diagnostic_task_basics(tmp_path):
    """Test DiagnosticTask basic attributes."""
    diag_script = tmp_path / 'diag_cow.py'
    tasks = _get_single_diagnostic_task(tmp_path, diag_script)
    task_names = [task.name for task in tasks]
    assert task_names == ['task0']
    ancestor_names = [anc.name for anc in list(tasks)[0].ancestors]
    assert ancestor_names == ['task0-ancestor0', 'task0-ancestor1']
    task = list(tasks)[0]
    assert task.script == diag_script
    assert task.settings == {'run_dir': tmp_path / 'mydiag' / 'run_dir',
                             'profile_diagnostic': False}
    assert task.output_dir == tmp_path / 'mydiag'


def test_diagnostic_diag_script_none(tmp_path):
    """Test case when diagnostic script doesnt exist."""
    diag_script = tmp_path / 'diag_cow.py'
    with pytest.raises(DiagnosticError) as err_msg:
        _get_single_diagnostic_task(tmp_path, diag_script, write_diag=False)
    diagnostics_root = os.path.join(DIAGNOSTICS_PATH, 'diag_scripts')
    script_file = os.path.abspath(os.path.join(diagnostics_root, diag_script))
    suf = "Cannot execute script "
    ept = suf + "'{}' ({}): file does not exist.".format(script_file,
                                                         script_file)
    assert ept == str(err_msg.value)


def _get_diagnostic_tasks(tmp_path, diagnostic_text, extension):
    """Assemble Python diagnostic tasks of DiagnosticTasks."""
    diag = 'diag_cow.' + extension
    diag_script = tmp_path / diag
    diag_output_dir = tmp_path / 'mydiag'
    diag_run_dir = diag_output_dir / 'run_dir'
    diag_plot_dir = diag_output_dir / 'plot_dir'
    diag_work_dir = diag_output_dir / 'work_dir'
    diag_settings = {'run_dir': diag_run_dir.as_posix(),
                     'plot_dir': diag_plot_dir.as_posix(),
                     'work_dir': diag_work_dir.as_posix(),
                     'profile_diagnostic': False,
                     'exit_on_ncl_warning': False}

    with open(diag_script, "w") as fil:
        fil.write(diagnostic_text)

    task = DiagnosticTask(
        name='task0',
        ancestors=None,
        script=diag_script.as_posix(),
        settings=diag_settings,
        output_dir=diag_output_dir.as_posix(),
    )

    return {task}


def test_py_diagnostic_run_sequential_task_fails(monkeypatch, tmp_path):
    """Run DiagnosticTask sequentially with bad Python script."""
    diagnostic_text = "import os\n\nprint(cow)"
    tasks = _get_diagnostic_tasks(tmp_path, diagnostic_text, 'py')

    def _run(self, input_filesi=[]):
        print(f'running task {self.name}')

    monkeypatch.setattr(BaseTask, '_run', _run)

    with pytest.raises(DiagnosticError) as err_mssg:
        _run_tasks_sequential(tasks)
    exp_mssg = "diag_cow.py failed with return code 1"
    assert exp_mssg in str(err_mssg.value)


def test_py_diagnostic_run_parallel_task_fails(monkeypatch, tmp_path):
    """Run DiagnosticTask parallel with bad Python script."""
    diagnostic_text = "import os\n\nprint(cow)"
    tasks = _get_diagnostic_tasks(tmp_path, diagnostic_text, 'py')

    def _run(self, input_filesi=[]):
        print(f'running task {self.name}')

    monkeypatch.setattr(BaseTask, '_run', _run)

    with pytest.raises(DiagnosticError) as err_mssg:
        _run_tasks_parallel(tasks, 2)
    exp_mssg = "diag_cow.py failed with return code 1"
    assert exp_mssg in str(err_mssg.value)


def test_py_diagnostic_run_parallel_task(monkeypatch, tmp_path):
    """Run DiagnosticTask in parallel with OK Python script."""
    diagnostic_text = "import os\n\nprint('cow')"
    tasks = _get_diagnostic_tasks(tmp_path, diagnostic_text, 'py')

    def _run(self, input_filesi=[]):
        print(f'running task {self.name}')

    monkeypatch.setattr(BaseTask, '_run', _run)

    _run_tasks_parallel(tasks, 2)


def test_ncl_diagnostic_run_parallel_task_fails(monkeypatch, tmp_path):
    """Run DiagnosticTask in parallel with OK NCL script."""
    diagnostic_text = "cows on the [river]"

    def _run(self, input_filesi=[]):
        print(f'running task {self.name}')

    if shutil.which('ncl') is not None:
        tasks = _get_diagnostic_tasks(tmp_path, diagnostic_text, 'ncl')

        monkeypatch.setattr(BaseTask, '_run', _run)
        with pytest.raises(DiagnosticError) as err_mssg:
            _run_tasks_parallel(tasks, 2)
        exp_mssg_1 = "An error occurred during execution of NCL script"
        exp_mssg_2 = "diag_cow.ncl"
        assert exp_mssg_1 in str(err_mssg.value)
        assert exp_mssg_2 in str(err_mssg.value)
    else:
        with pytest.raises(DiagnosticError) as err_mssg:
            tasks = _get_diagnostic_tasks(tmp_path, diagnostic_text, 'ncl')
        exp_mssg_1 = "Cannot execute script "
        exp_mssg_2 = "program 'ncl' not installed."
        assert exp_mssg_1 in str(err_mssg.value)
        assert exp_mssg_2 in str(err_mssg.value)


def test_ncl_diagnostic_run_parallel_task(monkeypatch, tmp_path):
    """Run DiagnosticTask in parallel with OK NCL script."""
    diagnostic_text = _py2ncl({'cow': 22}, 'tas')

    def _run(self, input_filesi=[]):
        print(f'running task {self.name}')

    if shutil.which('ncl') is not None:
        tasks = _get_diagnostic_tasks(tmp_path, diagnostic_text, 'ncl')
        monkeypatch.setattr(BaseTask, '_run', _run)
        _run_tasks_parallel(tasks, 2)
    else:
        with pytest.raises(DiagnosticError) as err_mssg:
            tasks = _get_diagnostic_tasks(tmp_path, diagnostic_text, 'ncl')
        exp_mssg_1 = "Cannot execute script "
        exp_mssg_2 = "program 'ncl' not installed."
        assert exp_mssg_1 in str(err_mssg.value)
        assert exp_mssg_2 in str(err_mssg.value)


def test_r_diagnostic_run_parallel_task(monkeypatch, tmp_path):
    """Run DiagnosticTask in parallel with OK NCL script."""
    diagnostic_text = 'var0 <- "zg"'

    def _run(self, input_filesi=[]):
        print(f'running task {self.name}')

    if shutil.which('Rscript') is not None:
        tasks = _get_diagnostic_tasks(tmp_path, diagnostic_text, 'R')
        monkeypatch.setattr(BaseTask, '_run', _run)
        _run_tasks_parallel(tasks, 2)
    else:
        with pytest.raises(DiagnosticError) as err_mssg:
            tasks = _get_diagnostic_tasks(tmp_path, diagnostic_text, 'ncl')
        exp_mssg_1 = "Cannot execute script "
        exp_mssg_2 = "program 'Rscript' not installed."
        assert exp_mssg_1 in str(err_mssg.value)
        assert exp_mssg_2 in str(err_mssg.value)
