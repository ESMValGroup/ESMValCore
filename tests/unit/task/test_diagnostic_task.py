import copy
import logging
import stat
from pathlib import Path

import pytest
import yaml

import esmvalcore._task
from esmvalcore._config._diagnostics import TagsManager
from esmvalcore._task import DiagnosticError, write_ncl_settings


def test_write_ncl_settings(tmp_path):
    """Test minimally write_ncl_settings()."""
    settings = {
        'run_dir': str(tmp_path / 'run_dir'),
        'diag_script_info': {'profile_diagnostic': False},
        'var_name': 'tas',
    }
    file_name = tmp_path / "settings"
    write_ncl_settings(settings, file_name)
    with open(file_name, 'r') as file:
        lines = file.readlines()
        assert 'var_name = "tas"\n' in lines
        assert 'if (isvar("profile_diagnostic")) then\n' not in lines

    settings = {
        'run_dir': str(tmp_path / 'run_dir'),
        'profile_diagnostic': True,
        'var_name': 'tas',
    }
    file_name = tmp_path / "settings"
    write_ncl_settings(settings, file_name)
    with open(file_name, 'r') as file:
        lines = file.readlines()
        assert 'var_name = "tas"\n' in lines
        assert 'profile_diagnostic' not in lines


@pytest.mark.parametrize("ext", ['.jl', '.py', '.ncl', '.R'])
def test_initialize_env(ext, tmp_path, monkeypatch):
    """Test that the environmental variables are set correctly."""
    monkeypatch.setattr(esmvalcore._task.DiagnosticTask, '_initialize_cmd',
                        lambda self: None)

    esmvaltool_path = tmp_path / 'esmvaltool'
    monkeypatch.setattr(esmvalcore._config.DIAGNOSTICS, 'path',
                        esmvaltool_path)

    diagnostics_path = esmvaltool_path / 'diag_scripts'
    diagnostics_path.mkdir(parents=True)
    script = diagnostics_path / ('test' + ext)
    script.touch()

    settings = {
        'run_dir': str(tmp_path / 'run_dir'),
        'profile_diagnostic': False,
    }
    task = esmvalcore._task.DiagnosticTask(
        script,
        settings,
        output_dir=str(tmp_path),
    )

    # Create correct environment
    env = {}
    test_env = copy.deepcopy(task.env)
    if ext in ('.jl', '.py'):
        env['MPLBACKEND'] = 'Agg'
    if ext == '.jl':
        env['JULIA_LOAD_PATH'] = f"{esmvaltool_path / 'install' / 'Julia'}"

        # check for new type of JULIA_LOAD_PATH
        # and cut away new path arguments @:@$CONDA_ENV:@stdlib
        # see https://github.com/ESMValGroup/ESMValCore/issues/1443
        test_env['JULIA_LOAD_PATH'] = \
            task.env['JULIA_LOAD_PATH'].split(":")[0]
    if ext in ('.ncl', '.R'):
        env['diag_scripts'] = str(diagnostics_path)

    assert test_env == env


CMD = {
    # ext, profile: expected command
    ('.py', False): ['python'],
    ('.py', True): ['python', '-m', 'vprof', '-o'],
    ('.ncl', False): ['ncl', '-n', '-p'],
    ('.ncl', True): ['ncl', '-n', '-p'],
    ('.R', False): ['Rscript'],
    ('.R', True): ['Rscript'],
    ('.jl', False): ['julia'],
    ('.jl', True): ['julia'],
    ('', False): [],
    ('', True): [],
}


@pytest.mark.parametrize("ext_profile,cmd", CMD.items())
def test_initialize_cmd(ext_profile, cmd, tmp_path, monkeypatch):
    """Test creating the command to run the diagnostic script."""
    monkeypatch.setattr(esmvalcore._task.DiagnosticTask, '_initialize_env',
                        lambda self: None)

    ext, profile = ext_profile
    script = tmp_path / ('test' + ext)
    script.touch()
    if ext == '':
        # test case where file is executable
        script.chmod(stat.S_IEXEC)

    run_dir = tmp_path / 'run_dir'
    settings = {
        'run_dir': str(run_dir),
        'profile_diagnostic': profile,
    }

    monkeypatch.setattr(esmvalcore._task, 'which', lambda x: x)
    monkeypatch.setattr(esmvalcore._task.sys, 'executable', 'python')

    task = esmvalcore._task.DiagnosticTask(script,
                                           settings,
                                           output_dir=str(tmp_path))

    # Append filenames to expected command
    if ext == '.py' and profile:
        cmd.append(str(run_dir / 'profile.json'))
        cmd.append('-c')
        cmd.append('c')
    cmd.append(str(script))
    assert task.cmd == cmd

    # test for no executable
    monkeypatch.setattr(esmvalcore._task, 'which', lambda x: None)
    if ext_profile[0] != '' and ext_profile[0] != '.py':
        with pytest.raises(DiagnosticError) as err_mssg:
            esmvalcore._task.DiagnosticTask(script,
                                            settings,
                                            output_dir=str(tmp_path))
        exp_mssg1 = "Cannot execute script "
        exp_mssg2 = "program '{}' not installed.".format(CMD[ext_profile][0])
        assert exp_mssg1 in str(err_mssg.value)
        assert exp_mssg2 in str(err_mssg.value)


@pytest.fixture
def diagnostic_task(mocker, tmp_path):
    class TrackedFile(esmvalcore._task.TrackedFile):
        provenance = None

    mocker.patch.object(esmvalcore._task, 'TrackedFile', autospec=TrackedFile)
    tags = TagsManager({'plot_type': {'tag': 'tag_value'}})
    mocker.patch.dict(esmvalcore._task.TAGS, tags)
    mocker.patch.object(esmvalcore._task,
                        '_write_citation_files',
                        autospec=True)

    mocker.patch.object(esmvalcore._task.DiagnosticTask, '_initialize_cmd')
    mocker.patch.object(esmvalcore._task.DiagnosticTask, '_initialize_env')

    settings = {
        'run_dir': str(tmp_path / 'run_dir'),
        'profile_diagnostic': False,
        'some_diagnostic_setting': True,
    }

    task = esmvalcore._task.DiagnosticTask('test.py',
                                           settings,
                                           output_dir=str(tmp_path),
                                           name='some-diagnostic-task')
    return task


def write_mock_provenance(diagnostic_task, record):
    run_dir = Path(diagnostic_task.settings['run_dir'])
    run_dir.mkdir(parents=True)
    provenance_file = run_dir / 'diagnostic_provenance.yml'
    provenance_file.write_text(yaml.safe_dump(record))


def test_collect_provenance(mocker, diagnostic_task):
    tracked_file_instance = mocker.Mock()
    tracked_file_class = mocker.patch.object(
        esmvalcore._task, 'TrackedFile', return_value=tracked_file_instance)
    write_citation = mocker.patch.object(esmvalcore._task,
                                         '_write_citation_files')

    record = {
        "test.png": {
            "caption": "Some figure",
            "ancestors": ["xyz.nc"],
            "plot_type": ["tag"],
        },
    }

    write_mock_provenance(diagnostic_task, record)

    ancestor_product = mocker.Mock()
    ancestor_product.filename = "xyz.nc"

    ancestor_task = mocker.Mock()
    ancestor_task.products = {ancestor_product}

    diagnostic_task.ancestors = [ancestor_task]

    diagnostic_task.products = mocker.Mock(autospec=set)
    diagnostic_task._collect_provenance()

    tracked_file_class.assert_called_once_with(
        "test.png",
        {
            "caption": "Some figure",
            "plot_type": ("tag_value", ),
            "script_file": "test.py",
            "some_diagnostic_setting": True,
        },
        {ancestor_product},
    )
    tracked_file_instance.initialize_provenance.assert_called_once_with(
        diagnostic_task.activity)
    tracked_file_instance.save_provenance.assert_called_once()
    write_citation.assert_called_once_with(tracked_file_instance.filename,
                                           tracked_file_instance.provenance)
    diagnostic_task.products.add.assert_called_once_with(tracked_file_instance)


def assert_warned(log, msgs):
    """Check that messages have been logged."""
    assert len(log.records) == len(msgs)
    for msg, record in zip(msgs, log.records):
        for snippet in msg:
            assert snippet in record.message


def test_collect_no_provenance(caplog, diagnostic_task):

    diagnostic_task._collect_provenance()
    assert_warned(caplog, [["No provenance information was written"]])


def test_collect_provenance_no_ancestors(caplog, diagnostic_task):

    caplog.set_level(logging.INFO)

    record = {
        "test.png": {
            "caption": "Some figure",
        },
    }

    write_mock_provenance(diagnostic_task, record)

    diagnostic_task._collect_provenance()

    assert_warned(caplog, [
        ["No ancestor files specified", "test.png"],
        ["Valid ancestor files"],
    ])


def test_collect_provenance_invalid_ancestors(caplog, diagnostic_task):

    caplog.set_level(logging.INFO)

    record = {
        "test.png": {
            "caption": "Some figure",
            "ancestors": ["xyz.nc"],
        },
    }

    write_mock_provenance(diagnostic_task, record)

    diagnostic_task._collect_provenance()

    assert_warned(caplog, [
        ["Invalid ancestor file", "test.png"],
        ["Valid ancestor files"],
    ])


def test_collect_provenance_ancestor_hint(mocker, caplog, diagnostic_task):

    caplog.set_level(logging.INFO)

    record = {
        "test.png": {
            "caption": "Some figure",
            "ancestors": ["xyz.nc"],
        },
        "test.nc": {
            "ancestors": ["abc.nc"],
        },
    }

    write_mock_provenance(diagnostic_task, record)

    ancestor_product = mocker.Mock()
    ancestor_product.filename = "xyz.nc"

    ancestor_task = mocker.Mock()
    ancestor_task.products = {ancestor_product}

    diagnostic_task.ancestors = [ancestor_task]
    diagnostic_task._collect_provenance()

    assert_warned(caplog, [
        ["Invalid ancestor file", "abc.nc", "test.nc"],
        ["Valid ancestor files", "xyz.nc"],
    ])
