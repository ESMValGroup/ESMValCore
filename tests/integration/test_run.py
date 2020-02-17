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


def check_log(tmp_path, pattern):

    log_dir = os.path.join(tmp_path, 'output_dir')
    log_file = os.path.join(log_dir,
                            os.listdir(log_dir)[0], 'run', 'main_log.txt')
    with open(log_file, 'r') as log:
        cases = []
        for line in log.readlines():
            case = pattern in line
            cases.append(case)
        assert True in cases


def check(tmp_path):
    """Check the run results."""
    log_dir = os.path.join(tmp_path, 'output_dir')
    log_file = os.path.join(log_dir,
                            os.listdir(log_dir)[0], 'run', 'main_log.txt')
    debug_file = os.path.join(log_dir,
                              os.listdir(log_dir)[0], 'run',
                              'main_log_debug.txt')
    assert os.listdir(log_dir)
    assert len(os.listdir(log_dir)) == 1
    recipe_dir = os.listdir(log_dir)[0]
    assert float(recipe_dir.split("_")[-1])
    assert float(recipe_dir.split("_")[-2])
    assert os.path.isfile(debug_file)
    assert os.path.isfile(log_file)


@contextlib.contextmanager
def arguments(*args):
    backup = sys.argv
    sys.argv = list(args)
    yield
    sys.argv = backup


def test_recipe_from_diags(tmp_path):

    recipe_file = "recipe_autoassess_stratosphere.yml"
    config_user_file = write_config_user_file(tmp_path)
    with arguments(
            'esmvaltool',
            '-c',
            config_user_file,
            str(recipe_file),
    ):
        with pytest.raises(SystemExit):
            run()


def test_no_recipe(tmp_path):

    recipe_file = "recipe_cows_and_oxen.yml"
    config_user_file = write_config_user_file(tmp_path)
    with arguments(
            'esmvaltool',
            '-c',
            config_user_file,
            str(recipe_file),
    ):
        with pytest.raises(SystemExit):
            run()
    check_log(tmp_path, "Specified recipe file does not exist")


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

    check(tmp_path)


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

    check_log(tmp_path, "Missing data")
    check(tmp_path)
