import stat

import pytest

import esmvalcore._task
from esmvalcore._task import DiagnosticError


@pytest.mark.parametrize("ext", ['.jl', '.py', '.ncl', '.R'])
def test_diagnostic_task_env(ext, tmp_path, monkeypatch):
    """Test that the environmental variables are set correctly."""
    monkeypatch.setattr(esmvalcore._task.DiagnosticTask, '_initialize_cmd',
                        lambda self: None)

    esmvaltool_path = tmp_path / 'esmvaltool'
    monkeypatch.setattr(esmvalcore._task, 'DIAGNOSTICS_PATH', esmvaltool_path)

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
    if ext in ('.jl', '.py'):
        env['MPLBACKEND'] = 'Agg'
    if ext == '.jl':
        env['JULIA_LOAD_PATH'] = f"{esmvaltool_path / 'install' / 'Julia'}:"
    if ext in ('.ncl', '.R'):
        env['diag_scripts'] = str(diagnostics_path)

    assert task.env == env


CMD = {
    # ext, profile: expected command
    ('.py', False): ['python'],
    ('.py', True): ['python', '-m', 'vmprof', '--lines', '-o'],
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
def test_diagnostic_task_cmd(ext_profile, cmd, tmp_path, monkeypatch):
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

    task = esmvalcore._task.DiagnosticTask(script,
                                           settings,
                                           output_dir=str(tmp_path))

    # Append filenames to expected command
    if ext == '.py' and profile:
        cmd.append(str(run_dir / 'profile.bin'))
    cmd.append(str(script))

    assert task.cmd == cmd

    # test for no executable
    monkeypatch.setattr(esmvalcore._task, 'which', lambda x: None)
    if ext_profile[0] != '':
        with pytest.raises(DiagnosticError) as err_mssg:
            task = esmvalcore._task.DiagnosticTask(script,
                                                   settings,
                                                   output_dir=str(tmp_path))
        exp_mssg1 = "Cannot execute script "
        exp_mssg2 = "program '{}' not installed.".format(CMD[ext_profile][0])
        assert exp_mssg1 in str(err_mssg.value)
        assert exp_mssg2 in str(err_mssg.value)
