"""Test that a task tree can be printed in a human readable form."""
import copy
import textwrap

import pytest

from esmvalcore._task import DiagnosticTask
from esmvalcore.preprocessor import PreprocessingTask, PreprocessorFile


@pytest.fixture
def preproc_file():
    return PreprocessorFile(
        attributes={'filename': '/output/preproc/file.nc'},
        settings={
            'extract_levels': {
                'scheme': 'linear',
                'levels': [95000]
            },
        },
    )


@pytest.fixture
def preproc_task(preproc_file):
    return PreprocessingTask(products=[preproc_file])


@pytest.fixture
def diagnostic_task(tmp_path):
    mock_script = tmp_path / 'script.py'
    mock_script.touch()
    settings = {
        'run_dir': str('/output/run'),
        'profile_diagnostic': False,
    }
    task = DiagnosticTask(mock_script, settings, output_dir='/output/run')
    task.script = '/some/where/esmvaltool/diag_scripts/test.py'
    return task


def test_repr_preproc_task(preproc_task):
    """Test printing a preprocessor task."""
    preproc_task.name = 'diag_1/tas'
    result = str(preproc_task)
    print(result)

    reference = textwrap.dedent("""
    PreprocessingTask: diag_1/tas
    order: ['extract_levels', 'save']
    PreprocessorFile: /output/preproc/file.nc
    {'extract_levels': {'levels': [95000], 'scheme': 'linear'},
     'save': {'filename': '/output/preproc/file.nc'}}
    ancestors:
    None
    """)

    assert result.strip() == reference.strip()


def test_repr_diagnostic_task(diagnostic_task):
    """Test printing a diagnostic task."""
    diagnostic_task.name = 'diag_1/script_1'
    result = str(diagnostic_task)
    print(result)

    reference = textwrap.dedent("""
    DiagnosticTask: diag_1/script_1
    script: /some/where/esmvaltool/diag_scripts/test.py
    settings:
    {'profile_diagnostic': False, 'run_dir': '/output/run'}
    ancestors:
    None
    """)

    assert result.strip() == reference.strip()


def test_repr_simple_tree(preproc_task, diagnostic_task):
    """Test the most common task tree."""
    preproc_task.name = 'diag_1/tas'
    diagnostic_task.name = 'diag_1/script_1'
    diagnostic_task.ancestors = [preproc_task]
    result = str(diagnostic_task)
    print(result)

    reference = textwrap.dedent("""
    DiagnosticTask: diag_1/script_1
    script: /some/where/esmvaltool/diag_scripts/test.py
    settings:
    {'profile_diagnostic': False, 'run_dir': '/output/run'}
    ancestors:
      PreprocessingTask: diag_1/tas
      order: ['extract_levels', 'save']
      PreprocessorFile: /output/preproc/file.nc
      {'extract_levels': {'levels': [95000], 'scheme': 'linear'},
       'save': {'filename': '/output/preproc/file.nc'}}
      ancestors:
      None
    """)

    assert result.strip() == reference.strip()


def test_repr_full_tree(preproc_task, diagnostic_task):
    """Test a more comlicated task tree."""
    derive_input_task_1 = copy.deepcopy(preproc_task)
    derive_input_task_1.name = 'diag_1/tas_derive_input_1'

    derive_input_task_2 = copy.deepcopy(preproc_task)
    derive_input_task_2.name = 'diag_1/tas_derive_input_2'

    preproc_task.name = 'diag_1/tas'
    preproc_task.ancestors = [derive_input_task_1, derive_input_task_2]

    diagnostic_task_1 = copy.deepcopy(diagnostic_task)
    diagnostic_task_1.name = 'diag_1/script_1'
    diagnostic_task_1.ancestors = [preproc_task]

    diagnostic_task.name = 'diag_1/script_2'
    diagnostic_task.ancestors = [diagnostic_task_1]
    result = str(diagnostic_task)
    print(result)

    reference = textwrap.dedent("""
    DiagnosticTask: diag_1/script_2
    script: /some/where/esmvaltool/diag_scripts/test.py
    settings:
    {'profile_diagnostic': False, 'run_dir': '/output/run'}
    ancestors:
      DiagnosticTask: diag_1/script_1
      script: /some/where/esmvaltool/diag_scripts/test.py
      settings:
      {'profile_diagnostic': False, 'run_dir': '/output/run'}
      ancestors:
        PreprocessingTask: diag_1/tas
        order: ['extract_levels', 'save']
        PreprocessorFile: /output/preproc/file.nc
        {'extract_levels': {'levels': [95000], 'scheme': 'linear'},
         'save': {'filename': '/output/preproc/file.nc'}}
        ancestors:
          PreprocessingTask: diag_1/tas_derive_input_1
          order: ['extract_levels', 'save']
          PreprocessorFile: /output/preproc/file.nc
          {'extract_levels': {'levels': [95000], 'scheme': 'linear'},
           'save': {'filename': '/output/preproc/file.nc'}}
          ancestors:
          None

          PreprocessingTask: diag_1/tas_derive_input_2
          order: ['extract_levels', 'save']
          PreprocessorFile: /output/preproc/file.nc
          {'extract_levels': {'levels': [95000], 'scheme': 'linear'},
           'save': {'filename': '/output/preproc/file.nc'}}
          ancestors:
          None
    """)

    assert result.strip() == reference.strip()
