"""Test diagnostic script runs."""
import contextlib
import os
import sys
from textwrap import dedent

import pytest
import yaml

from esmvalcore._main import run


def write_config_user_file(dirname, add_procs=False):
    config_file = dirname / 'config-user.yml'
    cfg = {
        'output_dir': str(dirname / 'output_dir'),
        'rootpath': {
            'default': str(dirname / 'input_dir'),
        },
        'drs': {
            'CMIP5': 'BADC',
        },
        'log_level': 'debug',
    }
    if add_procs:
        cfg["max_parallel_tasks"] = 2
    config_file.write_text(yaml.safe_dump(cfg, encoding=None))
    return str(config_file)


@contextlib.contextmanager
def arguments(*args):
    backup = sys.argv
    sys.argv = list(args)
    yield
    sys.argv = backup


def check(result_file):
    """Check the results."""
    result = yaml.safe_load(result_file.read_text())

    print(result)

    required_keys = {
        'input_files',
        'log_level',
        'plot_dir',
        'run_dir',
        'work_dir',
    }
    missing = required_keys - set(result)
    assert not missing


# python only for Core
SCRIPTS = {
    'diagnostic.py':
    dedent("""
        import yaml
        from esmvaltool.diag_scripts.shared import run_diagnostic

        def main(cfg):
            with open(cfg['setting_name'], 'w') as file:
                yaml.safe_dump(cfg, file)

        if __name__ == '__main__':
            with run_diagnostic() as config:
                main(config)
        """),
}


def test_diagnostic_null_with_no_processors(tmp_path):

    recipe_file = tmp_path / 'recipe_test_null.yml'

    # Create recipe
    recipe = dedent("""
        documentation:
          description: Recipe with no data.
          authors: [predoi_valeriu]

        diagnostics:
          diagnostic_name:
            scripts: null
        """)
    recipe_file.write_text(str(recipe))

    config_user_file = write_config_user_file(tmp_path)
    with arguments(
            'esmvaltool',
            '-c',
            config_user_file,
            str(recipe_file),
    ):
        with pytest.raises(SystemExit):
            run()


def test_diagnostic_null(tmp_path):

    recipe_file = tmp_path / 'recipe_test_null.yml'

    # Create recipe
    recipe = dedent("""
        documentation:
          description: Recipe with no data.
          authors: [predoi_valeriu]

        preprocessors:
          pp_rad:
            regrid:
              target_grid: 1x1
              scheme: linear

        diagnostics:
          diagnostic_name:
            description: "cows and oxen"
            variables:
              ta:
                mip: Amon
                preprocessor: pp_rad
              pr:
                mip: Amon
                preprocessor: pp_rad
            additional_datasets:
              - {dataset: CERES-EBAF,  project: obs4mips,
                 level: L3B,  version: Ed2-7,  start_year: 2001,
                 end_year: 2012, tier: 1}
            scripts: null
        """)
    recipe_file.write_text(str(recipe))

    config_user_file = write_config_user_file(tmp_path, add_procs=True)

    with arguments(
            'esmvaltool',
            '-c',
            config_user_file,
            str(recipe_file),
    ):
        with pytest.raises(SystemExit):
            run()

    log_dir = os.path.join(tmp_path, 'output_dir')
    log_file = os.path.join(log_dir,
                            os.listdir(log_dir)[0], 'run', 'main_log.txt')
    with open(log_file, 'r') as log:
        cases = []
        for line in log.readlines():
            case = "Missing data" in line
            cases.append(case)
        assert True in cases


@pytest.mark.parametrize('script_file, script', SCRIPTS.items())
def test_diagnostic_run(tmp_path, script_file, script):

    recipe_file = tmp_path / 'recipe_test.yml'
    script_file = tmp_path / script_file
    result_file = tmp_path / 'result.yml'

    # Write script to file
    script_file.write_text(str(script))

    # Create recipe
    recipe = dedent("""
        documentation:
          description: Recipe with no data.
          authors: [andela_bouwe]

        diagnostics:
          diagnostic_name:
            scripts:
              script_name:
                script: {}
                setting_name: {}
        """.format(script_file, result_file))
    recipe_file.write_text(str(recipe))

    config_user_file = write_config_user_file(tmp_path)
    with arguments(
            'esmvaltool',
            '-c',
            config_user_file,
            str(recipe_file),
    ):
        run()

    check(result_file)
