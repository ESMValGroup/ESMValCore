import os
from pathlib import Path
from pprint import pformat
from textwrap import dedent
from unittest.mock import create_autospec

import iris
import pytest
import yaml
from PIL import Image

import esmvalcore
from esmvalcore._config import TAGS
from esmvalcore._recipe import TASKSEP, read_recipe_file
from esmvalcore._task import DiagnosticTask
from esmvalcore.cmor.check import CheckLevels
from esmvalcore.exceptions import RecipeError
from esmvalcore.preprocessor import DEFAULT_ORDER, PreprocessingTask
from esmvalcore.preprocessor._io import concatenate_callback

from .test_diagnostic_run import write_config_user_file
from .test_provenance import check_provenance

TAGS_FOR_TESTING = {
    'authors': {
        'andela_bouwe': {
            'name': 'Bouwe, Andela',
        },
    },
    'projects': {
        'c3s-magic': 'C3S MAGIC project',
    },
    'themes': {
        'phys': 'physics',
    },
    'realms': {
        'atmos': 'atmosphere',
    },
    'statistics': {
        'mean': 'mean',
        'var': 'variability',
    },
    'domains': {
        'et': 'extra tropics',
        'trop': 'tropics',
    },
    'plot_types': {
        'zonal': 'zonal',
    },
}

MANDATORY_DATASET_KEYS = (
    'dataset',
    'diagnostic',
    'end_year',
    'filename',
    'frequency',
    'institute',
    'long_name',
    'mip',
    'modeling_realm',
    'preprocessor',
    'project',
    'short_name',
    'standard_name',
    'start_year',
    'units',
)

MANDATORY_SCRIPT_SETTINGS_KEYS = (
    'log_level',
    'script',
    'plot_dir',
    'run_dir',
    'work_dir',
)

DEFAULT_PREPROCESSOR_STEPS = (
    'add_fx_variables',
    'cleanup',
    'cmor_check_data',
    'cmor_check_metadata',
    'concatenate',
    'clip_start_end_year',
    'fix_data',
    'fix_file',
    'fix_metadata',
    'load',
    'remove_fx_variables',
    'save',
)

INITIALIZATION_ERROR_MSG = 'Could not create all tasks'


@pytest.fixture
def config_user(tmp_path):
    filename = write_config_user_file(tmp_path)
    cfg = esmvalcore._config.read_config_user_file(filename, 'recipe_test', {})
    cfg['offline'] = True
    cfg['check_level'] = CheckLevels.DEFAULT
    return cfg


def create_test_file(filename, tracking_id=None):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    attributes = {}
    if tracking_id is not None:
        attributes['tracking_id'] = tracking_id
    cube = iris.cube.Cube([], attributes=attributes)

    iris.save(cube, filename)


def _get_default_settings_for_chl(fix_dir, save_filename, preprocessor):
    """Get default preprocessor settings for chl."""
    standard_name = ('mass_concentration_of_phytoplankton_'
                     'expressed_as_chlorophyll_in_sea_water')
    defaults = {
        'load': {
            'callback': concatenate_callback,
        },
        'concatenate': {},
        'fix_file': {
            'alias': 'CanESM2',
            'dataset': 'CanESM2',
            'diagnostic': 'diagnostic_name',
            'end_year': 2005,
            'ensemble': 'r1i1p1',
            'exp': 'historical',
            'filename': fix_dir.replace('_fixed', '.nc'),
            'frequency': 'yr',
            'institute': ['CCCma'],
            'long_name': 'Total Chlorophyll Mass Concentration',
            'mip': 'Oyr',
            'modeling_realm': ['ocnBgchem'],
            'original_short_name': 'chl',
            'output_dir': fix_dir,
            'preprocessor': preprocessor,
            'product': 'output1',
            'project': 'CMIP5',
            'recipe_dataset_index': 0,
            'short_name': 'chl',
            'standard_name': standard_name,
            'start_year': 2000,
            'units': 'kg m-3',
            'variable_group': 'chl',
        },
        'fix_data': {
            'check_level': CheckLevels.DEFAULT,
            'alias': 'CanESM2',
            'dataset': 'CanESM2',
            'diagnostic': 'diagnostic_name',
            'end_year': 2005,
            'ensemble': 'r1i1p1',
            'exp': 'historical',
            'filename': fix_dir.replace('_fixed', '.nc'),
            'frequency': 'yr',
            'institute': ['CCCma'],
            'long_name': 'Total Chlorophyll Mass Concentration',
            'mip': 'Oyr',
            'modeling_realm': ['ocnBgchem'],
            'original_short_name': 'chl',
            'preprocessor': preprocessor,
            'product': 'output1',
            'project': 'CMIP5',
            'recipe_dataset_index': 0,
            'short_name': 'chl',
            'standard_name': standard_name,
            'start_year': 2000,
            'units': 'kg m-3',
            'variable_group': 'chl',
        },
        'fix_metadata': {
            'check_level': CheckLevels.DEFAULT,
            'alias': 'CanESM2',
            'dataset': 'CanESM2',
            'diagnostic': 'diagnostic_name',
            'end_year': 2005,
            'ensemble': 'r1i1p1',
            'exp': 'historical',
            'filename': fix_dir.replace('_fixed', '.nc'),
            'frequency': 'yr',
            'institute': ['CCCma'],
            'long_name': 'Total Chlorophyll Mass Concentration',
            'mip': 'Oyr',
            'modeling_realm': ['ocnBgchem'],
            'original_short_name': 'chl',
            'preprocessor': preprocessor,
            'product': 'output1',
            'project': 'CMIP5',
            'recipe_dataset_index': 0,
            'short_name': 'chl',
            'standard_name': standard_name,
            'start_year': 2000,
            'units': 'kg m-3',
            'variable_group': 'chl',
        },
        'clip_start_end_year': {
            'start_year': 2000,
            'end_year': 2005,
        },
        'cmor_check_metadata': {
            'check_level': CheckLevels.DEFAULT,
            'cmor_table': 'CMIP5',
            'mip': 'Oyr',
            'short_name': 'chl',
            'frequency': 'yr',
        },
        'cmor_check_data': {
            'check_level': CheckLevels.DEFAULT,
            'cmor_table': 'CMIP5',
            'mip': 'Oyr',
            'short_name': 'chl',
            'frequency': 'yr',
        },
        'add_fx_variables': {
            'fx_variables': {},
            'check_level': CheckLevels.DEFAULT,
        },
        'remove_fx_variables': {},
        'cleanup': {
            'remove': [fix_dir]
        },
        'save': {
            'compress': False,
            'filename': save_filename,
        }
    }
    return defaults


def _get_filenames(root_path, filenames, tracking_id):
    filename = filenames[0]
    filename = str(root_path / 'input' / filename)
    filenames = []
    if filename.endswith('[_.]*nc'):
        # Restore when we support filenames with no dates
        # filenames.append(filename.replace('[_.]*nc', '.nc'))
        filename = filename.replace('[_.]*nc', '_*.nc')
    if filename.endswith('*.nc'):
        filename = filename[:-len('*.nc')] + '_'
        intervals = [
            '1990_1999',
            '2000_2009',
            '2010_2019',
        ]
        for interval in intervals:
            filenames.append(filename + interval + '.nc')
    else:
        filenames.append(filename)

    for filename in filenames:
        create_test_file(filename, next(tracking_id))
    return filenames


@pytest.fixture
def patched_datafinder(tmp_path, monkeypatch):

    def tracking_ids(i=0):
        while True:
            yield i
            i += 1

    tracking_id = tracking_ids()

    def find_files(_, filenames):
        # Any occurrence of [something] in filename should have
        # been replaced before this function is called.
        for filename in filenames:
            assert '{' not in filename
        return _get_filenames(tmp_path, filenames, tracking_id)

    monkeypatch.setattr(esmvalcore._data_finder, 'find_files', find_files)


@pytest.fixture
def patched_failing_datafinder(tmp_path, monkeypatch):

    def tracking_ids(i=0):
        while True:
            yield i
            i += 1

    tracking_id = tracking_ids()

    def find_files(_, filenames):
        # Any occurrence of [something] in filename should have
        # been replaced before this function is called.
        for filename in filenames:
            assert '{' not in filename

        # Fail for specified fx variables
        for filename in filenames:
            if 'fx_' in filename:
                return []
            if 'sftlf' in filename:
                return []
            if 'IyrAnt_' in filename:
                return []
            if 'IyrGre_' in filename:
                return []
        return _get_filenames(tmp_path, filenames, tracking_id)

    monkeypatch.setattr(esmvalcore._data_finder, 'find_files', find_files)


@pytest.fixture
def patched_tas_derivation(monkeypatch):

    def get_required(short_name, _):
        if short_name != 'tas':
            assert False
        required = [
            {
                'short_name': 'pr'
            },
            {
                'short_name': 'areacella',
                'mip': 'fx',
                'optional': True
            },
        ]
        return required

    monkeypatch.setattr(esmvalcore._recipe, 'get_required', get_required)


DEFAULT_DOCUMENTATION = dedent("""
    documentation:
      title: Test recipe
      description: This is a test recipe.
      authors:
        - andela_bouwe
      references:
        - contact_authors
        - acknow_project
      projects:
        - c3s-magic
    """)


def get_recipe(tempdir, content, cfg):
    """Save and load recipe content."""
    recipe_file = tempdir / 'recipe_test.yml'
    # Add mandatory documentation section
    content = str(DEFAULT_DOCUMENTATION + content)
    recipe_file.write_text(content)

    recipe = read_recipe_file(str(recipe_file), cfg)

    return recipe


def test_recipe_no_datasets(tmp_path, config_user):
    content = dedent("""
        preprocessors:
          preprocessor_name:
            extract_levels:
              levels: 85000
              scheme: nearest

        diagnostics:
          diagnostic_name:
            variables:
              ta:
                preprocessor: preprocessor_name
                project: CMIP5
                mip: Amon
                exp: historical
                ensemble: r1i1p1
                start_year: 1999
                end_year: 2002
            scripts: null
        """)
    exc_message = ("You have not specified any dataset "
                   "or additional_dataset groups for variable "
                   "{'preprocessor': 'preprocessor_name', 'project': 'CMIP5',"
                   " 'mip': 'Amon', 'exp': 'historical', 'ensemble': 'r1i1p1'"
                   ", 'start_year': 1999, 'end_year': 2002, 'variable_group':"
                   " 'ta', 'short_name': 'ta', 'diagnostic': "
                   "'diagnostic_name'} Exiting.")
    with pytest.raises(RecipeError) as exc:
        get_recipe(tmp_path, content, config_user)
    assert str(exc.value) == exc_message


def test_simple_recipe(tmp_path, patched_datafinder, config_user):
    script = tmp_path / 'diagnostic.py'
    script.write_text('')
    content = dedent("""
        datasets:
          - dataset: bcc-csm1-1

        preprocessors:
          preprocessor_name:
            extract_levels:
              levels: 85000
              scheme: nearest

        diagnostics:
          diagnostic_name:
            additional_datasets:
              - dataset: GFDL-ESM2G
            variables:
              ta:
                preprocessor: preprocessor_name
                project: CMIP5
                mip: Amon
                exp: historical
                ensemble: r1i1p1
                start_year: 1999
                end_year: 2002
                additional_datasets:
                  - dataset: MPI-ESM-LR
            scripts:
              script_name:
                script: {}
                custom_setting: 1
        """.format(script))

    recipe = get_recipe(tmp_path, content, config_user)
    raw = yaml.safe_load(content)
    # Perform some sanity checks on recipe expansion/normalization
    print("Expanded recipe:")
    assert len(recipe.diagnostics) == len(raw['diagnostics'])
    for diagnostic_name, diagnostic in recipe.diagnostics.items():
        print(pformat(diagnostic))
        source = raw['diagnostics'][diagnostic_name]

        # Check that 'variables' have been read and updated
        assert len(diagnostic['preprocessor_output']) == len(
            source['variables'])
        for variable_name, variables in diagnostic[
                'preprocessor_output'].items():
            assert len(variables) == 3
            for variable in variables:
                for key in MANDATORY_DATASET_KEYS:
                    assert key in variable and variable[key]
                assert variable_name == variable['short_name']

    # Check that the correct tasks have been created
    variables = recipe.diagnostics['diagnostic_name']['preprocessor_output'][
        'ta']
    tasks = {t for task in recipe.tasks for t in task.flatten()}
    preproc_tasks = {t for t in tasks if isinstance(t, PreprocessingTask)}
    diagnostic_tasks = {t for t in tasks if isinstance(t, DiagnosticTask)}

    assert len(preproc_tasks) == 1
    for task in preproc_tasks:
        print("Task", task.name)
        assert task.order == list(DEFAULT_ORDER)
        for product in task.products:
            variable = [
                v for v in variables if v['filename'] == product.filename
            ][0]
            assert product.attributes == variable
            for step in DEFAULT_PREPROCESSOR_STEPS:
                assert step in product.settings
            assert len(product.files) == 2

    assert len(diagnostic_tasks) == 1
    for task in diagnostic_tasks:
        print("Task", task.name)
        assert task.ancestors == list(preproc_tasks)
        assert task.script == str(script)
        for key in MANDATORY_SCRIPT_SETTINGS_KEYS:
            assert key in task.settings and task.settings[key]
        assert task.settings['custom_setting'] == 1


def test_fx_preproc_error(tmp_path, patched_datafinder, config_user):
    script = tmp_path / 'diagnostic.py'
    script.write_text('')
    content = dedent("""
        datasets:
          - dataset: bcc-csm1-1

        preprocessors:
          preprocessor_name:
            extract_season:
              season: MAM

        diagnostics:
          diagnostic_name:
            variables:
              sftlf:
                preprocessor: preprocessor_name
                project: CMIP5
                mip: fx
                exp: historical
                ensemble: r0i0p0
                start_year: 1999
                end_year: 2002
                additional_datasets:
                  - dataset: MPI-ESM-LR
            scripts: null
        """)
    msg = ("Time coordinate preprocessor step(s) ['extract_season'] not "
           "permitted on fx vars, please remove them from recipe")
    with pytest.raises(Exception) as rec_err_exp:
        get_recipe(tmp_path, content, config_user)
    assert str(rec_err_exp.value) == INITIALIZATION_ERROR_MSG
    assert str(rec_err_exp.value.failed_tasks[0].message) == msg


def test_default_preprocessor(tmp_path, patched_datafinder, config_user):

    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              chl:
                project: CMIP5
                mip: Oyr
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: CanESM2}
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, config_user)

    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert len(task.products) == 1
    product = task.products.pop()
    preproc_dir = os.path.dirname(product.filename)
    assert preproc_dir.startswith(str(tmp_path))

    fix_dir = os.path.join(
        preproc_dir, 'CMIP5_CanESM2_Oyr_historical_r1i1p1_chl_2000-2005_fixed')
    defaults = _get_default_settings_for_chl(fix_dir, product.filename,
                                             'default')
    assert product.settings == defaults


def test_default_preprocessor_custom_order(tmp_path, patched_datafinder,
                                           config_user):
    """Test if default settings are used when ``custom_order`` is ``True``."""

    content = dedent("""
        preprocessors:
          default_custom_order:
            custom_order: true

        diagnostics:
          diagnostic_name:
            variables:
              chl:
                preprocessor: default_custom_order
                project: CMIP5
                mip: Oyr
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: CanESM2}
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, config_user)

    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert len(task.products) == 1
    product = task.products.pop()
    preproc_dir = os.path.dirname(product.filename)
    assert preproc_dir.startswith(str(tmp_path))

    fix_dir = os.path.join(
        preproc_dir, 'CMIP5_CanESM2_Oyr_historical_r1i1p1_chl_2000-2005_fixed')
    defaults = _get_default_settings_for_chl(fix_dir, product.filename,
                                             'default_custom_order')
    assert product.settings == defaults


def test_default_fx_preprocessor(tmp_path, patched_datafinder, config_user):

    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              sftlf:
                project: CMIP5
                mip: fx
                exp: historical
                ensemble: r0i0p0
                additional_datasets:
                  - {dataset: CanESM2}
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, config_user)

    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert len(task.products) == 1
    product = task.products.pop()
    preproc_dir = os.path.dirname(product.filename)
    assert preproc_dir.startswith(str(tmp_path))

    fix_dir = os.path.join(preproc_dir,
                           'CMIP5_CanESM2_fx_historical_r0i0p0_sftlf_fixed')

    defaults = {
        'load': {
            'callback': concatenate_callback,
        },
        'concatenate': {},
        'fix_file': {
            'alias': 'CanESM2',
            'dataset': 'CanESM2',
            'diagnostic': 'diagnostic_name',
            'ensemble': 'r0i0p0',
            'exp': 'historical',
            'filename': fix_dir.replace('_fixed', '.nc'),
            'frequency': 'fx',
            'institute': ['CCCma'],
            'long_name': 'Land Area Fraction',
            'mip': 'fx',
            'modeling_realm': ['atmos'],
            'original_short_name': 'sftlf',
            'output_dir': fix_dir,
            'preprocessor': 'default',
            'product': 'output1',
            'project': 'CMIP5',
            'recipe_dataset_index': 0,
            'short_name': 'sftlf',
            'standard_name': 'land_area_fraction',
            'units': '%',
            'variable_group': 'sftlf'
        },
        'fix_data': {
            'check_level': CheckLevels.DEFAULT,
            'alias': 'CanESM2',
            'dataset': 'CanESM2',
            'diagnostic': 'diagnostic_name',
            'ensemble': 'r0i0p0',
            'exp': 'historical',
            'filename': fix_dir.replace('_fixed', '.nc'),
            'frequency': 'fx',
            'institute': ['CCCma'],
            'long_name': 'Land Area Fraction',
            'mip': 'fx',
            'modeling_realm': ['atmos'],
            'original_short_name': 'sftlf',
            'preprocessor': 'default',
            'product': 'output1',
            'project': 'CMIP5',
            'recipe_dataset_index': 0,
            'short_name': 'sftlf',
            'standard_name': 'land_area_fraction',
            'units': '%',
            'variable_group': 'sftlf'
        },
        'fix_metadata': {
            'check_level': CheckLevels.DEFAULT,
            'alias': 'CanESM2',
            'dataset': 'CanESM2',
            'diagnostic': 'diagnostic_name',
            'ensemble': 'r0i0p0',
            'exp': 'historical',
            'filename': fix_dir.replace('_fixed', '.nc'),
            'frequency': 'fx',
            'institute': ['CCCma'],
            'long_name': 'Land Area Fraction',
            'mip': 'fx',
            'modeling_realm': ['atmos'],
            'original_short_name': 'sftlf',
            'preprocessor': 'default',
            'product': 'output1',
            'project': 'CMIP5',
            'recipe_dataset_index': 0,
            'short_name': 'sftlf',
            'standard_name': 'land_area_fraction',
            'units': '%',
            'variable_group': 'sftlf'
        },
        'cmor_check_metadata': {
            'check_level': CheckLevels.DEFAULT,
            'cmor_table': 'CMIP5',
            'mip': 'fx',
            'short_name': 'sftlf',
            'frequency': 'fx',
        },
        'cmor_check_data': {
            'check_level': CheckLevels.DEFAULT,
            'cmor_table': 'CMIP5',
            'mip': 'fx',
            'short_name': 'sftlf',
            'frequency': 'fx',
        },
        'add_fx_variables': {
            'fx_variables': {},
            'check_level': CheckLevels.DEFAULT,
        },
        'remove_fx_variables': {},
        'cleanup': {
            'remove': [fix_dir]
        },
        'save': {
            'compress': False,
            'filename': product.filename,
        }
    }
    assert product.settings == defaults


def test_empty_variable(tmp_path, patched_datafinder, config_user):
    """Test that it is possible to specify all information in the dataset."""
    content = dedent("""
        diagnostics:
          diagnostic_name:
            additional_datasets:
              - dataset: CanESM2
                project: CMIP5
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
            variables:
              pr:
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, config_user)
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert len(task.products) == 1
    product = task.products.pop()
    assert product.attributes['short_name'] == 'pr'
    assert product.attributes['dataset'] == 'CanESM2'


def test_cmip3_variable_autocomplete(tmp_path, patched_datafinder,
                                     config_user):
    """Test that required information is automatically added for CMIP5."""
    content = dedent("""
        diagnostics:
          test:
            additional_datasets:
              - dataset: bccr_bcm2_0
                project: CMIP3
                mip: A1
                frequency: mon
                exp: historical
                start_year: 2000
                end_year: 2001
                ensemble: r1i1p1
                modeling_realm: atmos
            variables:
              zg:
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, config_user)
    variable = recipe.diagnostics['test']['preprocessor_output']['zg'][0]

    reference = {
        'dataset': 'bccr_bcm2_0',
        'diagnostic': 'test',
        'end_year': 2001,
        'ensemble': 'r1i1p1',
        'exp': 'historical',
        'frequency': 'mon',
        'institute': ['BCCR'],
        'long_name': 'Geopotential Height',
        'mip': 'A1',
        'modeling_realm': 'atmos',
        'preprocessor': 'default',
        'project': 'CMIP3',
        'short_name': 'zg',
        'standard_name': 'geopotential_height',
        'start_year': 2000,
        'units': 'm',
    }
    for key in reference:
        assert variable[key] == reference[key]


def test_cmip5_variable_autocomplete(tmp_path, patched_datafinder,
                                     config_user):
    """Test that required information is automatically added for CMIP5."""
    content = dedent("""
        diagnostics:
          test:
            additional_datasets:
              - dataset: CanESM2
                project: CMIP5
                mip: 3hr
                exp: historical
                start_year: 2000
                end_year: 2001
                ensemble: r1i1p1
            variables:
              pr:
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, config_user)
    variable = recipe.diagnostics['test']['preprocessor_output']['pr'][0]

    reference = {
        'dataset': 'CanESM2',
        'diagnostic': 'test',
        'end_year': 2001,
        'ensemble': 'r1i1p1',
        'exp': 'historical',
        'frequency': '3hr',
        'institute': ['CCCma'],
        'long_name': 'Precipitation',
        'mip': '3hr',
        'modeling_realm': ['atmos'],
        'preprocessor': 'default',
        'project': 'CMIP5',
        'short_name': 'pr',
        'standard_name': 'precipitation_flux',
        'start_year': 2000,
        'units': 'kg m-2 s-1',
    }
    for key in reference:
        assert variable[key] == reference[key]


def test_cmip6_variable_autocomplete(tmp_path, patched_datafinder,
                                     config_user):
    """Test that required information is automatically added for CMIP6."""
    content = dedent("""
        diagnostics:
          test:
            additional_datasets:
              - dataset: HadGEM3-GC31-LL
                project: CMIP6
                mip: 3hr
                exp: historical
                start_year: 2000
                end_year: 2001
                ensemble: r2i1p1f1
                grid: gn
            variables:
              pr:
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, config_user)
    variable = recipe.diagnostics['test']['preprocessor_output']['pr'][0]

    reference = {
        'activity': 'CMIP',
        'dataset': 'HadGEM3-GC31-LL',
        'diagnostic': 'test',
        'end_year': 2001,
        'ensemble': 'r2i1p1f1',
        'exp': 'historical',
        'frequency': '3hr',
        'grid': 'gn',
        'institute': ['MOHC', 'NERC'],
        'long_name': 'Precipitation',
        'mip': '3hr',
        'modeling_realm': ['atmos'],
        'preprocessor': 'default',
        'project': 'CMIP6',
        'short_name': 'pr',
        'standard_name': 'precipitation_flux',
        'start_year': 2000,
        'units': 'kg m-2 s-1',
    }
    for key in reference:
        assert variable[key] == reference[key]


def test_simple_cordex_recipe(tmp_path, patched_datafinder, config_user):
    """Test simple CORDEX recipe."""
    content = dedent("""
        diagnostics:
          test:
            additional_datasets:
              - dataset: MOHC-HadGEM3-RA
                project: CORDEX
                product: output
                domain: AFR-44
                institute: MOHC
                driver: ECMWF-ERAINT
                exp: evaluation
                ensemble: r1i1p1
                rcm_version: v1
                start_year: 1991
                end_year: 1993
                mip: mon
            variables:
              tas:
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, config_user)
    variable = recipe.diagnostics['test']['preprocessor_output']['tas'][0]
    filename = variable.pop('filename').split('/')[-1]
    assert (filename ==
            'tas_MOHC-HadGEM3-RA_evaluation_r1i1p1_v1_mon_1991-1993.nc')
    reference = {
        'alias': 'MOHC-HadGEM3-RA',
        'dataset': 'MOHC-HadGEM3-RA',
        'diagnostic': 'test',
        'domain': 'AFR-44',
        'driver': 'ECMWF-ERAINT',
        'end_year': 1993,
        'ensemble': 'r1i1p1',
        'exp': 'evaluation',
        'frequency': 'mon',
        'institute': 'MOHC',
        'long_name': 'Near-Surface Air Temperature',
        'mip': 'mon',
        'modeling_realm': ['atmos'],
        'preprocessor': 'default',
        'product': 'output',
        'project': 'CORDEX',
        'recipe_dataset_index': 0,
        'rcm_version': 'v1',
        'short_name': 'tas',
        'original_short_name': 'tas',
        'standard_name': 'air_temperature',
        'start_year': 1991,
        'units': 'K',
        'variable_group': 'tas',
    }

    assert set(variable) == set(reference)
    for key in reference:
        assert variable[key] == reference[key]


def test_reference_dataset(tmp_path, patched_datafinder, config_user,
                           monkeypatch):

    levels = [100]
    get_reference_levels = create_autospec(
        esmvalcore._recipe.get_reference_levels, return_value=levels)
    monkeypatch.setattr(esmvalcore._recipe, 'get_reference_levels',
                        get_reference_levels)

    content = dedent("""
        preprocessors:
          test_from_reference:
            regrid:
              target_grid: reference_dataset
              scheme: linear
            extract_levels:
              levels: reference_dataset
              scheme: linear
          test_from_cmor_table:
            extract_levels:
              levels:
                cmor_table: CMIP6
                coordinate: alt16
              scheme: nearest

        diagnostics:
          diagnostic_name:
            variables:
              ta: &var
                preprocessor: test_from_reference
                project: CMIP5
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: GFDL-CM3}
                  - {dataset: MPI-ESM-LR}
                reference_dataset: MPI-ESM-LR
              ch4:
                <<: *var
                preprocessor: test_from_cmor_table
                additional_datasets:
                  - {dataset: GFDL-CM3}

            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, config_user)

    assert len(recipe.tasks) == 2

    # Check that the reference dataset has been used
    task = next(t for t in recipe.tasks
                if t.name == 'diagnostic_name' + TASKSEP + 'ta')
    assert len(task.products) == 2
    product = next(p for p in task.products
                   if p.attributes['dataset'] == 'GFDL-CM3')
    reference = next(p for p in task.products
                     if p.attributes['dataset'] == 'MPI-ESM-LR')

    assert product.settings['regrid']['target_grid'] == reference.files[0]
    assert product.settings['extract_levels']['levels'] == levels

    fix_dir = os.path.splitext(reference.filename)[0] + '_fixed'
    get_reference_levels.assert_called_once_with(
        filename=reference.files[0],
        project='CMIP5',
        dataset='MPI-ESM-LR',
        short_name='ta',
        mip='Amon',
        frequency='mon',
        fix_dir=fix_dir,
    )

    assert 'regrid' not in reference.settings
    assert 'extract_levels' not in reference.settings

    # Check that levels have been read from CMOR table
    task = next(t for t in recipe.tasks
                if t.name == 'diagnostic_name' + TASKSEP + 'ch4')
    assert len(task.products) == 1
    product = next(iter(task.products))
    assert product.settings['extract_levels']['levels'] == [
        0,
        250,
        750,
        1250,
        1750,
        2250,
        2750,
        3500,
        4500,
        6000,
        8000,
        10000,
        12000,
        14500,
        16000,
        18000,
    ]


def test_custom_preproc_order(tmp_path, patched_datafinder, config_user):

    content = dedent("""
        preprocessors:
          default: &default
            multi_model_statistics:
              span: overlap
              statistics: [mean ]
            area_statistics:
              operator: mean
          custom:
            custom_order: true
            <<: *default
          empty_custom:
            custom_order: true
          with_extract_time:
            custom_order: true
            extract_time:
              start_year: 2001
              start_month: 3
              start_day: 14
              end_year: 2002
              end_month: 6
              end_day: 28

        diagnostics:
          diagnostic_name:
            variables:
              chl_default: &chl
                short_name: chl
                preprocessor: default
                project: CMIP5
                mip: Oyr
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: CanESM2}
              chl_custom:
                <<: *chl
                preprocessor: custom
              chl_empty_custom:
                <<: *chl
                preprocessor: empty_custom
              chl_with_extract_time:
                <<: *chl
                preprocessor: with_extract_time
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, config_user)

    assert len(recipe.tasks) == 4

    for task in recipe.tasks:
        if task.name == 'diagnostic_name/chl_default':
            assert task.order.index('area_statistics') < task.order.index(
                'multi_model_statistics')
        elif task.name == 'diagnostic_name/chl_custom':
            assert task.order.index('area_statistics') > task.order.index(
                'multi_model_statistics')
        elif task.name == 'diagnostic_name/chl_empty_custom':
            assert len(task.products) == 1
            product = list(task.products)[0]
            assert set(
                product.settings.keys()) == set(DEFAULT_PREPROCESSOR_STEPS)
        elif task.name == 'diagnostic_name/chl_with_extract_time':
            assert len(task.products) == 1
            product = list(task.products)[0]
            steps = set(DEFAULT_PREPROCESSOR_STEPS + tuple(['extract_time']))
            assert set(product.settings.keys()) == steps
            assert product.settings['extract_time'] == {
                'start_year': 2001,
                'start_month': 3,
                'start_day': 14,
                'end_year': 2002,
                'end_month': 6,
                'end_day': 28,
            }
            assert product.settings['clip_start_end_year'] == {
                'start_year': 2000,
                'end_year': 2005,
            }
        else:
            assert False, f"invalid task {task.name}"


def test_derive(tmp_path, patched_datafinder, config_user):

    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              toz:
                project: CMIP5
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                derive: true
                force_derivation: true
                additional_datasets:
                  - {dataset: GFDL-CM3,  ensemble: r1i1p1}
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()

    assert task.name == 'diagnostic_name' + TASKSEP + 'toz'
    assert len(task.ancestors) == 2
    assert 'diagnostic_name' + TASKSEP + 'toz_derive_input_ps' in [
        t.name for t in task.ancestors
    ]
    assert 'diagnostic_name' + TASKSEP + 'toz_derive_input_tro3' in [
        t.name for t in task.ancestors
    ]

    # Check product content of tasks
    assert len(task.products) == 1
    product = task.products.pop()
    assert 'derive' in product.settings
    assert product.attributes['short_name'] == 'toz'
    assert product.files

    ps_product = next(p for a in task.ancestors for p in a.products
                      if p.attributes['short_name'] == 'ps')
    tro3_product = next(p for a in task.ancestors for p in a.products
                        if p.attributes['short_name'] == 'tro3')
    assert ps_product.filename in product.files
    assert tro3_product.filename in product.files


def test_derive_not_needed(tmp_path, patched_datafinder, config_user):

    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              toz:
                project: CMIP5
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                derive: true
                force_derivation: false
                additional_datasets:
                  - {dataset: GFDL-CM3,  ensemble: r1i1p1}
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()

    assert task.name == 'diagnostic_name/toz'
    assert len(task.ancestors) == 1
    ancestor = [t for t in task.ancestors][0]
    assert ancestor.name == 'diagnostic_name/toz_derive_input_toz'

    # Check product content of tasks
    assert len(task.products) == 1
    product = task.products.pop()
    assert product.attributes['short_name'] == 'toz'
    assert 'derive' in product.settings

    assert len(ancestor.products) == 1
    ancestor_product = ancestor.products.pop()
    assert ancestor_product.filename in product.files
    assert ancestor_product.attributes['short_name'] == 'toz'
    assert 'derive' not in ancestor_product.settings

    # Check that fixes are applied just once
    fixes = ('fix_file', 'fix_metadata', 'fix_data')
    for fix in fixes:
        assert fix in ancestor_product.settings
        assert fix not in product.settings


def test_derive_with_fx_ohc(tmp_path, patched_datafinder, config_user):
    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              ohc:
                mip: Omon
                exp: historical
                start_year: 2000
                end_year: 2005
                derive: true
                force_derivation: true
                additional_datasets:
                  - {dataset: GFDL-CM3, ensemble: r1i1p1,   project: CMIP5}
                  - {dataset: GFDL-CM4, ensemble: r1i1p1f1, project: CMIP6,
                     grid: gr1}
                  - {dataset: TEST, project: OBS, type: reanaly, version: 1,
                     tier: 1}

            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'ohc'

    # Check products
    all_product_files = []
    assert len(task.products) == 3
    for product in task.products:
        assert 'derive' in product.settings
        assert product.attributes['short_name'] == 'ohc'
        all_product_files.extend(product.files)

    # Check ancestors
    assert len(task.ancestors) == 2
    assert task.ancestors[0].name == (
        'diagnostic_name/ohc_derive_input_thetao')
    assert task.ancestors[1].name == (
        'diagnostic_name/ohc_derive_input_volcello')
    for ancestor_product in task.ancestors[0].products:
        assert ancestor_product.attributes['short_name'] == 'thetao'
        assert ancestor_product.filename in all_product_files
    for ancestor_product in task.ancestors[1].products:
        assert ancestor_product.attributes['short_name'] == 'volcello'
        if ancestor_product.attributes['project'] == 'CMIP6':
            assert ancestor_product.attributes['mip'] == 'Ofx'
        else:
            assert ancestor_product.attributes['mip'] == 'fx'
        assert ancestor_product.filename in all_product_files


def test_derive_with_fx_ohc_fail(tmp_path, patched_failing_datafinder,
                                 config_user):
    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              ohc:
                mip: Omon
                exp: historical
                start_year: 2000
                end_year: 2005
                derive: true
                force_derivation: true
                additional_datasets:
                  - {dataset: GFDL-CM3, ensemble: r1i1p1,   project: CMIP5}
                  - {dataset: GFDL-CM4, ensemble: r1i1p1f1, project: CMIP6,
                     grid: gr1}
                  - {dataset: TEST, project: OBS, type: reanaly, version: 1,
                     tier: 1}

            scripts: null
        """)
    with pytest.raises(RecipeError):
        get_recipe(tmp_path, content, config_user)


def test_derive_with_optional_var(tmp_path, patched_datafinder,
                                  patched_tas_derivation, config_user):
    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              tas:
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                derive: true
                force_derivation: true
                additional_datasets:
                  - {dataset: GFDL-CM3, ensemble: r1i1p1,   project: CMIP5}
                  - {dataset: GFDL-CM4, ensemble: r1i1p1f1, project: CMIP6,
                     grid: gr1}
                  - {dataset: TEST, project: OBS, type: reanaly, version: 1,
                     tier: 1}

            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'tas'

    # Check products
    all_product_files = []
    assert len(task.products) == 3
    for product in task.products:
        assert 'derive' in product.settings
        assert product.attributes['short_name'] == 'tas'
        all_product_files.extend(product.files)

    # Check ancestors
    assert len(task.ancestors) == 2
    assert task.ancestors[0].name == ('diagnostic_name/tas_derive_input_pr')
    assert task.ancestors[1].name == (
        'diagnostic_name/tas_derive_input_areacella')
    for ancestor_product in task.ancestors[0].products:
        assert ancestor_product.attributes['short_name'] == 'pr'
        assert ancestor_product.filename in all_product_files
    for ancestor_product in task.ancestors[1].products:
        assert ancestor_product.attributes['short_name'] == 'areacella'
        assert ancestor_product.filename in all_product_files


def test_derive_with_optional_var_nodata(tmp_path, patched_failing_datafinder,
                                         patched_tas_derivation, config_user):
    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              tas:
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                derive: true
                force_derivation: true
                additional_datasets:
                  - {dataset: GFDL-CM3, ensemble: r1i1p1,   project: CMIP5}
                  - {dataset: GFDL-CM4, ensemble: r1i1p1f1, project: CMIP6,
                     grid: gr1}
                  - {dataset: TEST, project: OBS, type: reanaly, version: 1,
                     tier: 1}

            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'tas'

    # Check products
    all_product_files = []
    assert len(task.products) == 3
    for product in task.products:
        assert 'derive' in product.settings
        assert product.attributes['short_name'] == 'tas'
        all_product_files.extend(product.files)

    # Check ancestors
    assert len(task.ancestors) == 1
    assert task.ancestors[0].name == ('diagnostic_name/tas_derive_input_pr')
    for ancestor_product in task.ancestors[0].products:
        assert ancestor_product.attributes['short_name'] == 'pr'
        assert ancestor_product.filename in all_product_files


def create_test_image(basename, cfg):
    """Get a valid path for saving a diagnostic plot."""
    image = Path(cfg['plot_dir']) / (basename + '.' + cfg['output_file_type'])
    image.parent.mkdir(parents=True)
    Image.new('RGB', (1, 1)).save(image)
    return str(image)


def get_diagnostic_filename(basename, cfg, extension='nc'):
    """Get a valid path for saving a diagnostic data file."""
    return os.path.join(
        cfg['work_dir'],
        basename + '.' + extension,
    )


def simulate_diagnostic_run(diagnostic_task):
    """Simulate Python diagnostic run."""
    cfg = diagnostic_task.settings
    input_files = [
        p.filename for a in diagnostic_task.ancestors for p in a.products
    ]
    record = {
        'caption': 'Test figure',
        'statistics': ['mean', 'var'],
        'domains': ['trop', 'et'],
        'plot_types': ['zonal'],
        'authors': ['andela_bouwe'],
        'references': ['acknow_project'],
        'ancestors': input_files,
    }

    diagnostic_file = get_diagnostic_filename('test', cfg)
    create_test_file(diagnostic_file)
    plot_file = create_test_image('test', cfg)
    provenance = os.path.join(cfg['run_dir'], 'diagnostic_provenance.yml')
    os.makedirs(cfg['run_dir'])
    with open(provenance, 'w') as file:
        yaml.safe_dump({diagnostic_file: record, plot_file: record}, file)

    diagnostic_task._collect_provenance()
    return record


def test_diagnostic_task_provenance(
    tmp_path,
    patched_datafinder,
    config_user,
):
    script = tmp_path / 'diagnostic.py'
    script.write_text('')

    TAGS.set_tag_values(TAGS_FOR_TESTING)

    content = dedent("""
        diagnostics:
          diagnostic_name:
            themes:
              - phys
            realms:
              - atmos
            variables:
              chl:
                project: CMIP5
                mip: Oyr
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - dataset: CanESM2
            scripts:
              script_name:
                script: {script}
              script_name2:
                script: {script}
                ancestors: [script_name]
        """.format(script=script))

    recipe = get_recipe(tmp_path, content, config_user)
    diagnostic_task = recipe.tasks.pop()

    simulate_diagnostic_run(next(iter(diagnostic_task.ancestors)))
    record = simulate_diagnostic_run(diagnostic_task)

    # Check resulting products
    assert len(diagnostic_task.products) == 2
    for product in diagnostic_task.products:
        product.restore_provenance()
        check_provenance(product)
        assert product.attributes['caption'] == record['caption']
        assert product.entity.get_attribute(
            'attribute:' + 'caption').pop() == record['caption']

        # Check that diagnostic script tags have been added
        for key in ('statistics', 'domains', 'authors'):
            assert product.attributes[key] == tuple(TAGS[key][k]
                                                    for k in record[key])

        # Check that recipe diagnostic tags have been added
        src = yaml.safe_load(DEFAULT_DOCUMENTATION + content)
        for key in ('realms', 'themes'):
            value = src['diagnostics']['diagnostic_name'][key]
            assert product.attributes[key] == tuple(TAGS[key][k]
                                                    for k in value)

        # Check that recipe tags have been added
        recipe_record = product.provenance.get_record('recipe:recipe_test.yml')
        assert len(recipe_record) == 1
        for key in ('description', 'references'):
            value = src['documentation'][key]
            if key == 'references':
                value = str(src['documentation'][key])
            assert recipe_record[0].get_attribute('attribute:' +
                                                  key).pop() == value

    # Test that provenance was saved to netcdf, xml and svg plot
    product = next(
        iter(p for p in diagnostic_task.products
             if p.filename.endswith('.nc')))
    cube = iris.load_cube(product.filename)
    assert cube.attributes['software'].startswith("Created with ESMValTool v")
    assert cube.attributes['caption'] == record['caption']
    prefix = os.path.splitext(product.filename)[0] + '_provenance'
    assert os.path.exists(prefix + '.xml')


def test_alias_generation(tmp_path, patched_datafinder, config_user):
    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              ta:
                project: CMIP5
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                grid: gn
                type: reanaly
                tier: 2
                version: latest
                additional_datasets:
                  - {dataset: GFDL-CM3,  ensemble: r1i1p1}
                  - {dataset: EC-EARTH,  ensemble: r1i1p1}
                  - {dataset: EC-EARTH,  ensemble: r2i1p1}
                  - {dataset: EC-EARTH,  ensemble: r3i1p1, alias: my_alias}
                  - {project: OBS, dataset: ERA-Interim,  version: 1}
                  - {project: OBS, dataset: ERA-Interim,  version: 2}
                  - {project: CMIP6, activity: CMP, dataset: GF3, ensemble: r1}
                  - {project: CMIP6, activity: CMP, dataset: GF2, ensemble: r1}
                  - {project: CMIP6, activity: HRMP, dataset: EC, ensemble: r1}
                  - {project: CMIP6, activity: HRMP, dataset: HA, ensemble: r1}
            scripts: null
        """)  # noqa:

    recipe = get_recipe(tmp_path, content, config_user)
    assert len(recipe.diagnostics) == 1
    diag = recipe.diagnostics['diagnostic_name']
    var = diag['preprocessor_output']['ta']
    for dataset in var:
        if dataset['project'] == 'CMIP5':
            if dataset['dataset'] == 'GFDL-CM3':
                assert dataset['alias'] == 'CMIP5_GFDL-CM3'
            else:
                if dataset['ensemble'] == 'r1i1p1':
                    assert dataset['alias'] == 'CMIP5_EC-EARTH_r1i1p1'
                elif dataset['ensemble'] == 'r2i1p1':
                    assert dataset['alias'] == 'CMIP5_EC-EARTH_r2i1p1'
                else:
                    assert dataset['alias'] == 'my_alias'
        elif dataset['project'] == 'CMIP6':
            if dataset['dataset'] == 'GF3':
                assert dataset['alias'] == 'CMIP6_CMP_GF3'
            elif dataset['dataset'] == 'GF2':
                assert dataset['alias'] == 'CMIP6_CMP_GF2'
            elif dataset['dataset'] == 'EC':
                assert dataset['alias'] == 'CMIP6_HRMP_EC'
            else:
                assert dataset['alias'] == 'CMIP6_HRMP_HA'
        else:
            if dataset['version'] == 1:
                assert dataset['alias'] == 'OBS_1'
            else:
                assert dataset['alias'] == 'OBS_2'


def test_concatenation(tmp_path, patched_datafinder, config_user):
    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              ta:
                project: CMIP5
                mip: Amon
                start_year: 2000
                end_year: 2005
                grid: gn
                type: reanaly
                tier: 2
                version: latest
                additional_datasets:
                  - dataset: GFDL-CM3
                    ensemble: r1i1p1
                    exp: [historical, rcp85]
                  - dataset: GFDL-CM3
                    ensemble: r1i1p1
                    exp: historical
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, config_user)
    assert len(recipe.diagnostics) == 1
    diag = recipe.diagnostics['diagnostic_name']
    var = diag['preprocessor_output']['ta']
    for dataset in var:
        if dataset['exp'] == 'historical':
            assert dataset['alias'] == 'historical'
        else:
            assert dataset['alias'] == 'historical-rcp85'


def test_ensemble_expansion(tmp_path, patched_datafinder, config_user):
    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              ta:
                project: CMIP5
                mip: Amon
                exp: historical
                ensemble: r(1:3)i1p1
                start_year: 2000
                end_year: 2005
                grid: gn
                type: reanaly
                tier: 2
                version: latest
                additional_datasets:
                  - {dataset: GFDL-CM3}
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, config_user)
    assert len(recipe.diagnostics) == 1
    diag = recipe.diagnostics['diagnostic_name']
    var = diag['preprocessor_output']['ta']
    assert len(var) == 3
    assert var[0]['ensemble'] == 'r1i1p1'
    assert var[1]['ensemble'] == 'r2i1p1'
    assert var[2]['ensemble'] == 'r3i1p1'


def test_extract_shape(tmp_path, patched_datafinder, config_user):
    TAGS.set_tag_values(TAGS_FOR_TESTING)

    content = dedent("""
        preprocessors:
          test:
            extract_shape:
              shapefile: test.shp

        diagnostics:
          test:
            variables:
              ta:
                preprocessor: test
                project: CMIP5
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: GFDL-CM3}
            scripts: null
        """)
    # Create shapefile
    shapefile = config_user['auxiliary_data_dir'] / Path('test.shp')
    shapefile.parent.mkdir(parents=True, exist_ok=True)
    shapefile.touch()

    recipe = get_recipe(tmp_path, content, config_user)

    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert len(task.products) == 1
    product = task.products.pop()
    assert product.settings['extract_shape']['shapefile'] == str(shapefile)


@pytest.mark.parametrize('invalid_arg',
                         ['shapefile', 'method', 'crop', 'decomposed'])
def test_extract_shape_raises(tmp_path, patched_datafinder, config_user,
                              invalid_arg):
    TAGS.set_tag_values(TAGS_FOR_TESTING)

    # Create shapefile
    shapefile = config_user['auxiliary_data_dir'] / Path('test.shp')
    shapefile.parent.mkdir(parents=True, exist_ok=True)
    shapefile.touch()

    content = dedent("""
        preprocessors:
          test:
            extract_shape:
              crop: true
              method: contains
              shapefile: test.shp

        diagnostics:
          test:
            variables:
              ta:
                preprocessor: test
                project: CMIP5
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  -
                      dataset: GFDL-CM3
            scripts: null
        """)

    # Add invalid argument
    recipe = yaml.safe_load(content)
    recipe['preprocessors']['test']['extract_shape'][invalid_arg] = 'x'
    content = yaml.safe_dump(recipe)

    with pytest.raises(RecipeError) as exc:
        get_recipe(tmp_path, content, config_user)

    assert str(exc.value) == INITIALIZATION_ERROR_MSG
    assert 'extract_shape' in exc.value.failed_tasks[0].message
    assert invalid_arg in exc.value.failed_tasks[0].message


def test_weighting_landsea_fraction(tmp_path, patched_datafinder, config_user):
    TAGS.set_tag_values(TAGS_FOR_TESTING)

    content = dedent("""
        preprocessors:
          landfrac_weighting:
            weighting_landsea_fraction:
              area_type: land

        diagnostics:
          diagnostic_name:
            variables:
              gpp:
                preprocessor: landfrac_weighting
                project: CMIP5
                mip: Lmon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: CanESM2}
                  - {dataset: TEST, project: obs4MIPs, level: 1, version: 1,
                     tier: 1}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'gpp'

    # Check weighting
    assert len(task.products) == 2
    for product in task.products:
        assert 'weighting_landsea_fraction' in product.settings
        settings = product.settings['weighting_landsea_fraction']
        assert len(settings) == 1
        assert settings['area_type'] == 'land'
        fx_variables = product.settings['add_fx_variables']['fx_variables']
        assert isinstance(fx_variables, dict)
        if product.attributes['project'] == 'obs4MIPs':
            assert len(fx_variables) == 1
            assert fx_variables.get('sftlf')
        else:
            assert len(fx_variables) == 2
            assert fx_variables.get('sftlf')
            assert fx_variables.get('sftof')


def test_weighting_landsea_fraction_no_fx(tmp_path, patched_failing_datafinder,
                                          config_user):
    content = dedent("""
        preprocessors:
          landfrac_weighting:
            weighting_landsea_fraction:
              area_type: land
              exclude: ['CMIP4-Model']

        diagnostics:
          diagnostic_name:
            variables:
              gpp:
                preprocessor: landfrac_weighting
                project: CMIP5
                mip: Lmon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: CanESM2}
                  - {dataset: TEST, project: obs4MIPs, level: 1, version: 1,
                     tier: 1}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'gpp'

    # Check weighting
    assert len(task.products) == 2
    for product in task.products:
        assert 'weighting_landsea_fraction' in product.settings
        settings = product.settings['weighting_landsea_fraction']
        assert len(settings) == 1
        assert 'exclude' not in settings
        assert settings['area_type'] == 'land'
        fx_variables = product.settings['add_fx_variables']['fx_variables']
        assert isinstance(fx_variables, dict)
        assert len(fx_variables) == 0


def test_weighting_landsea_fraction_exclude(tmp_path, patched_datafinder,
                                            config_user):
    content = dedent("""
        preprocessors:
          landfrac_weighting:
            weighting_landsea_fraction:
              area_type: land
              exclude: ['CanESM2', 'reference_dataset']

        diagnostics:
          diagnostic_name:
            variables:
              gpp:
                preprocessor: landfrac_weighting
                project: CMIP5
                mip: Lmon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                reference_dataset: GFDL-CM3
                additional_datasets:
                  - {dataset: CanESM2}
                  - {dataset: GFDL-CM3}
                  - {dataset: TEST, project: obs4MIPs, level: 1, version: 1,
                     tier: 1}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'gpp'

    # Check weighting
    assert len(task.products) == 3
    for product in task.products:
        if product.attributes['dataset'] != 'TEST':
            assert 'weighting_landsea_fraction' not in product.settings
            continue
        assert 'weighting_landsea_fraction' in product.settings
        settings = product.settings['weighting_landsea_fraction']
        assert len(settings) == 1
        assert 'exclude' not in settings
        assert settings['area_type'] == 'land'


def test_weighting_landsea_fraction_exclude_fail(tmp_path, patched_datafinder,
                                                 config_user):
    content = dedent("""
        preprocessors:
          landfrac_weighting:
            weighting_landsea_fraction:
              area_type: land
              exclude: ['alternative_dataset']

        diagnostics:
          diagnostic_name:
            variables:
              gpp:
                preprocessor: landfrac_weighting
                project: CMIP5
                mip: Lmon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                reference_dataset: GFDL-CM3
                additional_datasets:
                  - {dataset: CanESM2}
                  - {dataset: GFDL-CM3}
            scripts: null
        """)
    with pytest.raises(RecipeError) as exc_info:
        get_recipe(tmp_path, content, config_user)
    assert str(exc_info.value) == INITIALIZATION_ERROR_MSG
    assert str(exc_info.value.failed_tasks[0].message) == (
        'Preprocessor landfrac_weighting uses alternative_dataset, but '
        'alternative_dataset is not defined for variable gpp of diagnostic '
        'diagnostic_name')


def test_area_statistics(tmp_path, patched_datafinder, config_user):
    content = dedent("""
        preprocessors:
          area_statistics:
            area_statistics:
              operator: mean

        diagnostics:
          diagnostic_name:
            variables:
              gpp:
                preprocessor: area_statistics
                project: CMIP5
                mip: Lmon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: CanESM2}
                  - {dataset: TEST, project: obs4MIPs, level: 1, version: 1,
                     tier: 1}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'gpp'

    # Check area_statistics
    assert len(task.products) == 2
    for product in task.products:
        assert 'area_statistics' in product.settings
        settings = product.settings['area_statistics']
        assert len(settings) == 1
        assert settings['operator'] == 'mean'
        fx_variables = product.settings['add_fx_variables']['fx_variables']
        assert isinstance(fx_variables, dict)
        if product.attributes['project'] == 'obs4MIPs':
            assert len(fx_variables) == 1
            assert fx_variables.get('areacella')
        else:
            assert len(fx_variables) == 2
            assert fx_variables.get('areacella')
            assert fx_variables.get('areacello')


def test_landmask(tmp_path, patched_datafinder, config_user):
    content = dedent("""
        preprocessors:
          landmask:
            mask_landsea:
              mask_out: sea

        diagnostics:
          diagnostic_name:
            variables:
              gpp:
                preprocessor: landmask
                project: CMIP5
                mip: Lmon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: CanESM2}
                  - {dataset: TEST, project: obs4MIPs, level: 1, version: 1,
                     tier: 1}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'gpp'

    # Check weighting
    assert len(task.products) == 2
    for product in task.products:
        assert 'mask_landsea' in product.settings
        settings = product.settings['mask_landsea']
        assert len(settings) == 1
        assert settings['mask_out'] == 'sea'
        fx_variables = product.settings['add_fx_variables']['fx_variables']
        assert isinstance(fx_variables, dict)
        fx_variables = fx_variables.values()
        if product.attributes['project'] == 'obs4MIPs':
            assert len(fx_variables) == 1
        else:
            assert len(fx_variables) == 2


def test_user_defined_fxvar(tmp_path, patched_datafinder, config_user):
    content = dedent("""
        preprocessors:
          landmask:
            mask_landsea:
              mask_out: sea
              fx_variables:
                sftlf:
                  exp: piControl
            mask_landseaice:
              mask_out: sea
              fx_variables:
                sftgif:
                  exp: piControl
            volume_statistics:
              operator: mean
            area_statistics:
              operator: mean
              fx_variables:
                areacello:
                  mip: fx
                  exp: piControl
        diagnostics:
          diagnostic_name:
            variables:
              gpp:
                preprocessor: landmask
                project: CMIP5
                mip: Lmon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: CanESM2}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check custom fx variables
    task = recipe.tasks.pop()
    product = task.products.pop()

    # landsea
    settings = product.settings['mask_landsea']
    assert len(settings) == 1
    assert settings['mask_out'] == 'sea'
    fx_variables = product.settings['add_fx_variables']['fx_variables']
    assert isinstance(fx_variables, dict)
    assert len(fx_variables) == 4
    assert '_fx_' in fx_variables['sftlf']['filename']
    assert '_piControl_' in fx_variables['sftlf']['filename']

    # landseaice
    settings = product.settings['mask_landseaice']
    assert len(settings) == 1
    assert settings['mask_out'] == 'sea'
    assert '_fx_' in fx_variables['sftgif']['filename']
    assert '_piControl_' in fx_variables['sftgif']['filename']

    # volume statistics
    settings = product.settings['volume_statistics']
    assert len(settings) == 1
    assert settings['operator'] == 'mean'
    assert 'volcello' in fx_variables

    # area statistics
    settings = product.settings['area_statistics']
    assert len(settings) == 1
    assert settings['operator'] == 'mean'
    assert '_fx_' in fx_variables['areacello']['filename']
    assert '_piControl_' in fx_variables['areacello']['filename']


def test_user_defined_fxlist(tmp_path, patched_datafinder, config_user):
    content = dedent("""
        preprocessors:
          landmask:
            mask_landsea:
              mask_out: sea
              fx_variables: [{'short_name': 'sftlf', 'exp': 'piControl'}]
            mask_landseaice:
              mask_out: sea
              fx_variables: [{'short_name': 'sftgif', 'exp': 'piControl'}]
            volume_statistics:
              operator: mean
            area_statistics:
              operator: mean
              fx_variables: [{'short_name': 'areacello', 'mip': 'fx',
                         'exp': 'piControl'}]
        diagnostics:
          diagnostic_name:
            variables:
              gpp:
                preprocessor: landmask
                project: CMIP5
                mip: Lmon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: CanESM2}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check custom fx variables
    task = recipe.tasks.pop()
    product = task.products.pop()

    # landsea
    settings = product.settings['mask_landsea']
    assert len(settings) == 1
    assert settings['mask_out'] == 'sea'
    fx_variables = product.settings['add_fx_variables']['fx_variables']
    assert isinstance(fx_variables, dict)
    assert len(fx_variables) == 4
    assert '_fx_' in fx_variables['sftlf']['filename']
    assert '_piControl_' in fx_variables['sftlf']['filename']

    # landseaice
    settings = product.settings['mask_landseaice']
    assert len(settings) == 1
    assert settings['mask_out'] == 'sea'
    assert '_fx_' in fx_variables['sftlf']['filename']
    assert '_piControl_' in fx_variables['sftlf']['filename']

    # volume statistics
    settings = product.settings['volume_statistics']
    assert len(settings) == 1
    assert settings['operator'] == 'mean'
    assert 'volcello' in fx_variables

    # area statistics
    settings = product.settings['area_statistics']
    assert len(settings) == 1
    assert settings['operator'] == 'mean'
    assert '_fx_' in fx_variables['areacello']['filename']
    assert '_piControl_' in fx_variables['areacello']['filename']


def test_landmask_no_fx(tmp_path, patched_failing_datafinder, config_user):
    content = dedent("""
        preprocessors:
          landmask:
            mask_landsea:
              mask_out: sea
              always_use_ne_mask: false

        diagnostics:
          diagnostic_name:
            variables:
              gpp:
                preprocessor: landmask
                project: CMIP5
                mip: Lmon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: CanESM2}
                  - {dataset: CanESM5, project: CMIP6, grid: gn,
                     ensemble: r1i1p1f1}
                  - {dataset: TEST, project: obs4MIPs, level: 1, version: 1,
                     tier: 1}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'gpp'

    # Check masking
    assert len(task.products) == 3
    for product in task.products:
        assert 'mask_landsea' in product.settings
        settings = product.settings['mask_landsea']
        assert len(settings) == 2
        assert settings['mask_out'] == 'sea'
        assert settings['always_use_ne_mask'] is False
        fx_variables = product.settings['add_fx_variables']['fx_variables']
        assert isinstance(fx_variables, dict)
        fx_variables = fx_variables.values()
        assert not any(fx_variables)


def test_fx_vars_fixed_mip_cmip6(tmp_path, patched_datafinder, config_user):
    """Test fx variables with given mips."""
    TAGS.set_tag_values(TAGS_FOR_TESTING)

    content = dedent("""
        preprocessors:
          preproc:
           area_statistics:
             operator: mean
             fx_variables:
               sftgif:
                 mip: fx
               volcello:
                 ensemble: r2i1p1f1
                 mip: Ofx

        diagnostics:
          diagnostic_name:
            variables:
              tas:
                preprocessor: preproc
                project: CMIP6
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'tas'
    assert len(task.products) == 1
    product = task.products.pop()

    # Check area_statistics
    assert 'area_statistics' in product.settings
    settings = product.settings['area_statistics']
    assert len(settings) == 1
    assert settings['operator'] == 'mean'

    # Check add_fx_variables
    fx_variables = product.settings['add_fx_variables']['fx_variables']
    assert isinstance(fx_variables, dict)
    assert len(fx_variables) == 2
    assert '_fx_' in fx_variables['sftgif']['filename']
    assert '_r2i1p1f1_' in fx_variables['volcello']['filename']
    assert '_Ofx_' in fx_variables['volcello']['filename']


def test_fx_vars_invalid_mip_cmip6(tmp_path, patched_datafinder, config_user):
    """Test fx variables with invalid mip."""
    TAGS.set_tag_values(TAGS_FOR_TESTING)

    content = dedent("""
        preprocessors:
          preproc:
           area_statistics:
             operator: mean
             fx_variables:
               areacella:
                 mip: INVALID

        diagnostics:
          diagnostic_name:
            variables:
              tas:
                preprocessor: preproc
                project: CMIP6
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
            scripts: null
        """)
    msg = ("Requested mip table 'INVALID' for fx variable 'areacella' not "
           "available for project 'CMIP6'")
    with pytest.raises(RecipeError) as rec_err_exp:
        get_recipe(tmp_path, content, config_user)
    assert str(rec_err_exp.value) == INITIALIZATION_ERROR_MSG
    assert msg in rec_err_exp.value.failed_tasks[0].message


def test_fx_vars_invalid_mip_for_var_cmip6(tmp_path, patched_datafinder,
                                           config_user):
    """Test fx variables with invalid mip for variable."""
    TAGS.set_tag_values(TAGS_FOR_TESTING)

    content = dedent("""
        preprocessors:
          preproc:
           area_statistics:
             operator: mean
             fx_variables:
               areacella:
                 mip: Lmon

        diagnostics:
          diagnostic_name:
            variables:
              tas:
                preprocessor: preproc
                project: CMIP6
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
            scripts: null
        """)
    msg = ("fx variable 'areacella' not available in CMOR table 'Lmon' for "
           "'CMIP6'")
    with pytest.raises(RecipeError) as rec_err_exp:
        get_recipe(tmp_path, content, config_user)
    assert str(rec_err_exp.value) == INITIALIZATION_ERROR_MSG
    assert msg in rec_err_exp.value.failed_tasks[0].message


def test_fx_vars_mip_search_cmip6(tmp_path, patched_datafinder, config_user):
    """Test mip tables search for different fx variables."""
    TAGS.set_tag_values(TAGS_FOR_TESTING)

    content = dedent("""
        preprocessors:
          preproc:
           area_statistics:
             operator: mean
             fx_variables:
               areacella:
               areacello:
               clayfrac:
           mask_landsea:
             mask_out: sea

        diagnostics:
          diagnostic_name:
            variables:
              tas:
                preprocessor: preproc
                project: CMIP6
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'tas'
    assert len(task.products) == 1
    product = task.products.pop()

    # Check area_statistics
    assert 'area_statistics' in product.settings
    settings = product.settings['area_statistics']
    assert len(settings) == 1
    assert settings['operator'] == 'mean'

    # Check mask_landsea
    assert 'mask_landsea' in product.settings
    settings = product.settings['mask_landsea']
    assert len(settings) == 1
    assert settings['mask_out'] == 'sea'

    # Check add_fx_variables
    fx_variables = product.settings['add_fx_variables']['fx_variables']
    assert isinstance(fx_variables, dict)
    assert len(fx_variables) == 5
    assert '_fx_' in fx_variables['areacella']['filename']
    assert '_Ofx_' in fx_variables['areacello']['filename']
    assert '_Efx_' in fx_variables['clayfrac']['filename']
    assert '_fx_' in fx_variables['sftlf']['filename']
    assert '_Ofx_' in fx_variables['sftof']['filename']


def test_fx_list_mip_search_cmip6(tmp_path, patched_datafinder, config_user):
    """Test mip tables search for list of different fx variables."""
    content = dedent("""
        preprocessors:
          preproc:
           area_statistics:
             operator: mean
             fx_variables: [
               'areacella',
               'areacello',
               'clayfrac',
               'sftlf',
               'sftof',
               ]

        diagnostics:
          diagnostic_name:
            variables:
              tas:
                preprocessor: preproc
                project: CMIP6
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'tas'
    assert len(task.products) == 1
    product = task.products.pop()

    # Check area_statistics
    assert 'area_statistics' in product.settings
    settings = product.settings['area_statistics']
    assert len(settings) == 1
    assert settings['operator'] == 'mean'

    # Check add_fx_variables
    fx_variables = product.settings['add_fx_variables']['fx_variables']
    assert isinstance(fx_variables, dict)
    assert len(fx_variables) == 5
    assert '_fx_' in fx_variables['areacella']['filename']
    assert '_Ofx_' in fx_variables['areacello']['filename']
    assert '_Efx_' in fx_variables['clayfrac']['filename']
    assert '_fx_' in fx_variables['sftlf']['filename']
    assert '_Ofx_' in fx_variables['sftof']['filename']


def test_fx_vars_volcello_in_ofx_cmip6(tmp_path, patched_datafinder,
                                       config_user):
    TAGS.set_tag_values(TAGS_FOR_TESTING)

    content = dedent("""
        preprocessors:
          preproc:
           volume_statistics:
             operator: mean
             fx_variables:
               volcello:
                 mip: Ofx

        diagnostics:
          diagnostic_name:
            variables:
              tos:
                preprocessor: preproc
                project: CMIP6
                mip: Omon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'tos'
    assert len(task.products) == 1
    product = task.products.pop()

    # Check volume_statistics
    assert 'volume_statistics' in product.settings
    settings = product.settings['volume_statistics']
    assert len(settings) == 1
    assert settings['operator'] == 'mean'
    fx_variables = product.settings['add_fx_variables']['fx_variables']
    assert isinstance(fx_variables, dict)
    assert len(fx_variables) == 1
    assert '_Omon_' not in fx_variables['volcello']['filename']
    assert '_Ofx_' in fx_variables['volcello']['filename']


def test_fx_dicts_volcello_in_ofx_cmip6(tmp_path, patched_datafinder,
                                        config_user):
    content = dedent("""
        preprocessors:
          preproc:
           volume_statistics:
             operator: mean
             fx_variables:
               volcello:
                 mip: Oyr
                 exp: piControl

        diagnostics:
          diagnostic_name:
            variables:
              tos:
                preprocessor: preproc
                project: CMIP6
                mip: Omon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'tos'
    assert len(task.products) == 1
    product = task.products.pop()

    # Check volume_statistics
    assert 'volume_statistics' in product.settings
    settings = product.settings['volume_statistics']
    assert len(settings) == 1
    assert settings['operator'] == 'mean'
    fx_variables = product.settings['add_fx_variables']['fx_variables']
    assert isinstance(fx_variables, dict)
    assert len(fx_variables) == 1
    assert '_Oyr_' in fx_variables['volcello']['filename'][0]
    assert '_piControl_' in fx_variables['volcello']['filename'][0]
    assert '_Omon_' not in fx_variables['volcello']['filename'][0]


def test_fx_vars_list_no_preproc_cmip6(tmp_path, patched_datafinder,
                                       config_user):
    content = dedent("""
        preprocessors:
          preproc:
           regrid:
             target_grid: 1x1
             scheme: linear
           extract_volume:
             z_min: 0
             z_max: 100
           annual_statistics:
             operator: mean
           convert_units:
             units: K
           area_statistics:
             operator: mean

        diagnostics:
          diagnostic_name:
            variables:
              tos:
                preprocessor: preproc
                project: CMIP6
                mip: Omon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'tos'
    assert len(task.ancestors) == 0
    assert len(task.products) == 1
    product = task.products.pop()
    assert product.attributes['short_name'] == 'tos'
    assert product.files
    assert 'area_statistics' in product.settings
    settings = product.settings['area_statistics']
    assert len(settings) == 1
    assert settings['operator'] == 'mean'
    fx_variables = product.settings['add_fx_variables']['fx_variables']
    assert len(fx_variables) == 2


def test_fx_vars_volcello_in_omon_cmip6(tmp_path, patched_failing_datafinder,
                                        config_user):
    content = dedent("""
        preprocessors:
          preproc:
           volume_statistics:
             operator: mean
             fx_variables:
               volcello:
                 mip: Omon

        diagnostics:
          diagnostic_name:
            variables:
              tos:
                preprocessor: preproc
                project: CMIP6
                mip: Omon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'tos'
    assert len(task.products) == 1
    product = task.products.pop()

    # Check volume_statistics
    assert 'volume_statistics' in product.settings
    settings = product.settings['volume_statistics']
    assert len(settings) == 1
    assert settings['operator'] == 'mean'
    fx_variables = product.settings['add_fx_variables']['fx_variables']
    assert isinstance(fx_variables, dict)
    assert len(fx_variables) == 1
    assert '_Ofx_' not in fx_variables['volcello']['filename'][0]
    assert '_Omon_' in fx_variables['volcello']['filename'][0]


def test_fx_vars_volcello_in_oyr_cmip6(tmp_path, patched_failing_datafinder,
                                       config_user):
    content = dedent("""
        preprocessors:
          preproc:
           volume_statistics:
             operator: mean
             fx_variables:
               volcello:
                 mip: Oyr

        diagnostics:
          diagnostic_name:
            variables:
              o2:
                preprocessor: preproc
                project: CMIP6
                mip: Oyr
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'o2'
    assert len(task.products) == 1
    product = task.products.pop()

    # Check volume_statistics
    assert 'volume_statistics' in product.settings
    settings = product.settings['volume_statistics']
    assert len(settings) == 1
    assert settings['operator'] == 'mean'
    fx_variables = product.settings['add_fx_variables']['fx_variables']
    assert isinstance(fx_variables, dict)
    assert len(fx_variables) == 1
    assert '_Ofx_' not in fx_variables['volcello']['filename'][0]
    assert '_Oyr_' in fx_variables['volcello']['filename'][0]


def test_fx_vars_volcello_in_fx_cmip5(tmp_path, patched_datafinder,
                                      config_user):
    content = dedent("""
        preprocessors:
          preproc:
           volume_statistics:
             operator: mean
             fx_variables:
               volcello:

        diagnostics:
          diagnostic_name:
            variables:
              tos:
                preprocessor: preproc
                project: CMIP5
                mip: Omon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: CanESM2}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'tos'
    assert len(task.products) == 1
    product = task.products.pop()

    # Check volume_statistics
    assert 'volume_statistics' in product.settings
    settings = product.settings['volume_statistics']
    assert len(settings) == 1
    assert settings['operator'] == 'mean'
    fx_variables = product.settings['add_fx_variables']['fx_variables']
    assert isinstance(fx_variables, dict)
    assert len(fx_variables) == 1
    assert '_fx_' in fx_variables['volcello']['filename']
    assert '_Omon_' not in fx_variables['volcello']['filename']


def test_wrong_project(tmp_path, patched_datafinder, config_user):
    content = dedent("""
        preprocessors:
          preproc:
           volume_statistics:
             operator: mean
             fx_variables:
               volcello:

        diagnostics:
          diagnostic_name:
            variables:
              tos:
                preprocessor: preproc
                project: CMIP7
                mip: Omon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: CanESM2}
            scripts: null
        """)
    msg = ("Unable to load CMOR table (project) 'CMIP7' for variable 'tos' "
           "with mip 'Omon'")
    with pytest.raises(RecipeError) as wrong_proj:
        get_recipe(tmp_path, content, config_user)
    assert str(wrong_proj.value) == INITIALIZATION_ERROR_MSG
    assert str(wrong_proj.value.failed_tasks[0].message) == msg


def test_invalid_fx_var_cmip6(tmp_path, patched_datafinder, config_user):
    """Test that error is raised for invalid fx variable."""
    TAGS.set_tag_values(TAGS_FOR_TESTING)

    content = dedent("""
        preprocessors:
          preproc:
           area_statistics:
             operator: mean
             fx_variables:
               areacella:
               wrong_fx_variable:

        diagnostics:
          diagnostic_name:
            variables:
              tas:
                preprocessor: preproc
                project: CMIP6
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
            scripts: null
        """)
    msg = ("Requested fx variable 'wrong_fx_variable' not available in any "
           "CMOR table")
    with pytest.raises(RecipeError) as rec_err_exp:
        get_recipe(tmp_path, content, config_user)
    assert str(rec_err_exp.value) == INITIALIZATION_ERROR_MSG
    assert msg in rec_err_exp.value.failed_tasks[0].message


def test_ambiguous_fx_var_cmip6(tmp_path, patched_datafinder, config_user):
    """Test that error is raised for fx files available in multiple mips."""
    TAGS.set_tag_values(TAGS_FOR_TESTING)

    content = dedent("""
        preprocessors:
          preproc:
           area_statistics:
             operator: mean
             fx_variables:
               volcello:

        diagnostics:
          diagnostic_name:
            variables:
              tas:
                preprocessor: preproc
                project: CMIP6
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
            scripts: null
        """)
    msg = ("Requested fx variable 'volcello' for dataset 'CanESM5' of project "
           "'CMIP6' is available in more than one CMOR table for 'CMIP6': "
           "['Odec', 'Ofx', 'Omon', 'Oyr']")
    with pytest.raises(RecipeError) as rec_err_exp:
        get_recipe(tmp_path, content, config_user)
    assert str(rec_err_exp.value) == INITIALIZATION_ERROR_MSG
    assert msg in rec_err_exp.value.failed_tasks[0].message


def test_unique_fx_var_in_multiple_mips_cmip6(tmp_path,
                                              patched_failing_datafinder,
                                              config_user):
    """Test that no error is raised for fx files available in one mip."""
    TAGS.set_tag_values(TAGS_FOR_TESTING)

    content = dedent("""
        preprocessors:
          preproc:
           area_statistics:
             operator: mean
             fx_variables:
               sftgif:

        diagnostics:
          diagnostic_name:
            variables:
              tas:
                preprocessor: preproc
                project: CMIP6
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'tas'
    assert len(task.products) == 1
    product = task.products.pop()

    # Check area_statistics
    assert 'area_statistics' in product.settings
    settings = product.settings['area_statistics']
    assert len(settings) == 1
    assert settings['operator'] == 'mean'

    # Check add_fx_variables
    # Due to failing datafinder, only files in LImon are found even though
    # sftgif is available in the tables fx, IyrAnt, IyrGre and LImon
    fx_variables = product.settings['add_fx_variables']['fx_variables']
    assert isinstance(fx_variables, dict)
    assert len(fx_variables) == 1
    sftgif_files = fx_variables['sftgif']['filename']
    assert isinstance(sftgif_files, list)
    assert len(sftgif_files) == 1
    assert '_LImon_' in sftgif_files[0]


def test_multimodel_mask(tmp_path, patched_datafinder, config_user):
    """Test ``mask_multimodel``."""
    content = dedent("""
        preprocessors:
          preproc:
            mask_multimodel:

        diagnostics:
          diagnostic_name:
            variables:
              tas:
                preprocessor: preproc
                project: CMIP5
                mip: Amon
                exp: historical
                start_year: 2005
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: BNU-ESM}
                  - {dataset: CanESM2}
                  - {dataset: HadGEM2-ES}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == f'diagnostic_name{TASKSEP}tas'

    # Check mask_multimodel
    assert len(task.products) == 3
    for product in task.products:
        assert 'mask_multimodel' in product.settings
        assert product.settings['mask_multimodel'] == {}


def test_obs4mips_case_correct(tmp_path, patched_datafinder, config_user):
    """Test that obs4mips is corrected to obs4MIPs."""
    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              tas:
                project: CMIP5
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: TEST, project: obs4mips,
                     version: 1, tier: 1, level: 1}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, config_user)
    variable = recipe.diagnostics['diagnostic_name']['preprocessor_output'][
        'tas'][0]
    assert variable['project'] == 'obs4MIPs'


def test_recipe_run(tmp_path, patched_datafinder, config_user, mocker):

    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              areacella:
                project: CMIP5
                mip: fx
                exp: historical
                ensemble: r1i1p1
                additional_datasets:
                  - {dataset: BNU-ESM}
            scripts: null
        """)
    config_user['download_dir'] = tmp_path / 'download_dir'
    config_user['offline'] = False

    mocker.patch.object(esmvalcore._recipe.esgf,
                        'download',
                        create_autospec=True)

    recipe = get_recipe(tmp_path, content, config_user)

    recipe.tasks.run = mocker.Mock()
    recipe.run()

    esmvalcore._recipe.esgf.download.assert_called_once_with(
        set(), config_user['download_dir'])
    recipe.tasks.run.assert_called_once_with(
        max_parallel_tasks=config_user['max_parallel_tasks'])
