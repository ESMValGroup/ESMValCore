import os
from pathlib import Path
from pprint import pformat
from textwrap import dedent

import iris
import pytest
import yaml
from mock import create_autospec

import esmvalcore
from esmvalcore._recipe import TASKSEP, read_recipe_file
from esmvalcore._recipe_checks import RecipeError
from esmvalcore._task import DiagnosticTask
from esmvalcore.preprocessor import DEFAULT_ORDER, PreprocessingTask
from esmvalcore.preprocessor._io import concatenate_callback

from .test_diagnostic_run import write_config_user_file
from .test_provenance import check_provenance

MANDATORY_DATASET_KEYS = (
    'cmor_table',
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
    'cleanup',
    'cmor_check_data',
    'cmor_check_metadata',
    'concatenate',
    'extract_time',
    'fix_data',
    'fix_file',
    'fix_metadata',
    'load',
    'save',
)


@pytest.fixture
def config_user(tmp_path):
    filename = write_config_user_file(tmp_path)
    cfg = esmvalcore._config.read_config_user_file(filename, 'recipe_test')
    cfg['synda_download'] = False
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

        # Fail for fx variables
        for filename in filenames:
            if 'fx_' in filename:
                return []
        return _get_filenames(tmp_path, filenames, tracking_id)

    monkeypatch.setattr(esmvalcore._data_finder, 'find_files', find_files)


DEFAULT_DOCUMENTATION = dedent("""
    documentation:
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
    rec_err = "Time coordinate preprocessor step extract_season \
              not permitted on fx vars \
              please remove them from recipe."
    with pytest.raises(Exception) as rec_err_exp:
        get_recipe(tmp_path, content, config_user)
        assert rec_err == rec_err_exp


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
    defaults = {
        'load': {
            'callback': concatenate_callback,
        },
        'concatenate': {},
        'fix_file': {
            'project': 'CMIP5',
            'dataset': 'CanESM2',
            'short_name': 'chl',
            'output_dir': fix_dir,
        },
        'fix_data': {
            'project': 'CMIP5',
            'dataset': 'CanESM2',
            'short_name': 'chl',
            'cmor_table': 'CMIP5',
            'mip': 'Oyr',
            'frequency': 'yr',
        },
        'fix_metadata': {
            'project': 'CMIP5',
            'dataset': 'CanESM2',
            'short_name': 'chl',
            'cmor_table': 'CMIP5',
            'mip': 'Oyr',
            'frequency': 'yr',
        },
        'extract_time': {
            'start_year': 2000,
            'end_year': 2006,
            'start_month': 1,
            'end_month': 1,
            'start_day': 1,
            'end_day': 1,
        },
        'cmor_check_metadata': {
            'cmor_table': 'CMIP5',
            'mip': 'Oyr',
            'short_name': 'chl',
            'frequency': 'yr',
        },
        'cmor_check_data': {
            'cmor_table': 'CMIP5',
            'mip': 'Oyr',
            'short_name': 'chl',
            'frequency': 'yr',
        },
        'cleanup': {
            'remove': [fix_dir]
        },
        'save': {
            'compress': False,
            'filename': product.filename,
        }
    }
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
                start_year: 2000
                end_year: 2005
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
        preproc_dir,
        'CMIP5_CanESM2_fx_historical_r0i0p0_sftlf_2000-2005_fixed')

    defaults = {
        'load': {
            'callback': concatenate_callback,
        },
        'concatenate': {},
        'fix_file': {
            'project': 'CMIP5',
            'dataset': 'CanESM2',
            'short_name': 'sftlf',
            'output_dir': fix_dir,
        },
        'fix_data': {
            'project': 'CMIP5',
            'dataset': 'CanESM2',
            'short_name': 'sftlf',
            'cmor_table': 'CMIP5',
            'mip': 'fx',
            'frequency': 'fx',
        },
        'fix_metadata': {
            'project': 'CMIP5',
            'dataset': 'CanESM2',
            'short_name': 'sftlf',
            'cmor_table': 'CMIP5',
            'mip': 'fx',
            'frequency': 'fx',
        },
        'cmor_check_metadata': {
            'cmor_table': 'CMIP5',
            'mip': 'fx',
            'short_name': 'sftlf',
            'frequency': 'fx',
        },
        'cmor_check_data': {
            'cmor_table': 'CMIP5',
            'mip': 'fx',
            'short_name': 'sftlf',
            'frequency': 'fx',
        },
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
        'institute': ['MOHC'],
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
        reference.files[0],
        'CMIP5',
        'MPI-ESM-LR',
        'ta',
        fix_dir,
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
            area_statistics:
              operator: mean
            multi_model_statistics:
              span: overlap
              statistics: [mean ]
          custom:
            custom_order: true
            <<: *default

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
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, config_user)

    assert len(recipe.tasks) == 2

    default = next(t for t in recipe.tasks if tuple(t.order) == DEFAULT_ORDER)
    custom = next(t for t in recipe.tasks if tuple(t.order) != DEFAULT_ORDER)

    assert custom.order.index('area_statistics') < custom.order.index(
        'multi_model_statistics')
    assert default.order.index('area_statistics') > default.order.index(
        'multi_model_statistics')


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


def test_derive_with_fx_ohc_fail(tmp_path,
                                 patched_failing_datafinder,
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


def test_derive_with_fx_nbp_grid(tmp_path,
                                 patched_failing_datafinder,
                                 config_user):
    """The fx variable needed for nbp_grid is declared as 'optional'."""
    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              nbp_grid:
                mip: Lmon
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
    assert task.name == 'diagnostic_name' + TASKSEP + 'nbp_grid'

    # Check products
    all_product_files = []
    assert len(task.products) == 3
    for product in task.products:
        assert 'derive' in product.settings
        assert product.attributes['short_name'] == 'nbp_grid'
        all_product_files.extend(product.files)

    # Check ancestors
    assert len(task.ancestors) == 1
    assert task.ancestors[0].name == (
        'diagnostic_name/nbp_grid_derive_input_nbp')
    for ancestor_product in task.ancestors[0].products:
        assert ancestor_product.attributes['short_name'] == 'nbp'
        assert ancestor_product.filename in all_product_files


def get_plot_filename(basename, cfg):
    """Get a valid path for saving a diagnostic plot."""
    return os.path.join(
        cfg['plot_dir'],
        basename + '.' + cfg['output_file_type'],
    )


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
        'caption': 'Test plot',
        'plot_file': get_plot_filename('test', cfg),
        'statistics': ['mean', 'var'],
        'domains': ['trop', 'et'],
        'plot_types': ['zonal'],
        'authors': ['andela_bouwe'],
        'references': ['acknow_project'],
        'ancestors': input_files,
    }

    diagnostic_file = get_diagnostic_filename('test', cfg)
    create_test_file(diagnostic_file)
    provenance = os.path.join(cfg['run_dir'], 'diagnostic_provenance.yml')
    os.makedirs(cfg['run_dir'])
    with open(provenance, 'w') as file:
        yaml.safe_dump({diagnostic_file: record}, file)

    diagnostic_task._collect_provenance()
    return record


TAGS = {
    'authors': {
        'andela_bouwe': {
            'name': 'Bouwe Andela',
        },
    },
    'references': {
        'acknow_author': "Please acknowledge the author(s).",
        'contact_authors': "Please contact the author(s) ...",
        'acknow_project': "Please acknowledge the project(s).",
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


def test_diagnostic_task_provenance(
        tmp_path,
        patched_datafinder,
        monkeypatch,
        config_user,
):
    monkeypatch.setattr(esmvalcore._config, 'TAGS', TAGS)
    monkeypatch.setattr(esmvalcore._recipe, 'TAGS', TAGS)
    monkeypatch.setattr(esmvalcore._task, 'TAGS', TAGS)

    script = tmp_path / 'diagnostic.py'
    script.write_text('')

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

    # Check resulting product
    product = diagnostic_task.products.pop()
    check_provenance(product)
    for key in ('caption', 'plot_file'):
        assert product.attributes[key] == record[key]
        assert product.entity.get_attribute('attribute:' +
                                            key).pop() == record[key]

    # Check that diagnostic script tags have been added
    for key in ('statistics', 'domains', 'authors', 'references'):
        assert product.attributes[key] == tuple(TAGS[key][k]
                                                for k in record[key])

    # Check that recipe diagnostic tags have been added
    src = yaml.safe_load(DEFAULT_DOCUMENTATION + content)
    for key in ('realms', 'themes'):
        value = src['diagnostics']['diagnostic_name'][key]
        assert product.attributes[key] == tuple(TAGS[key][k] for k in value)

    # Check that recipe tags have been added
    recipe_record = product.provenance.get_record('recipe:recipe_test.yml')
    assert len(recipe_record) == 1
    for key in ('description', 'references'):
        value = src['documentation'][key]
        if key == 'references':
            value = ', '.join(TAGS[key][k] for k in value)
        assert recipe_record[0].get_attribute('attribute:' +
                                              key).pop() == value

    # Test that provenance was saved to netcdf, xml and svg plot
    cube = iris.load(product.filename)[0]
    assert 'provenance' in cube.attributes
    prefix = os.path.splitext(product.filename)[0] + '_provenance'
    assert os.path.exists(prefix + '.xml')
    assert os.path.exists(prefix + '.svg')


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
                  - {dataset: EC-EARTH,  ensemble: r3i1p1, alias: custom_alias}
                  - {project: OBS, dataset: ERA-Interim,  version: 1}
                  - {project: OBS, dataset: ERA-Interim,  version: 2}
                  - {project: CMIP6, dataset: GFDL-CM3,  ensemble: r1i1p1}
                  - {project: CMIP6, dataset: EC-EARTH,  ensemble: r1i1p1}
                  - {project: CMIP6, dataset: HADGEM,  ensemble: r1i1p1}
            scripts: null
        """)

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
                    assert dataset['alias'] == 'custom_alias'
        elif dataset['project'] == 'CMIP6':
            if dataset['dataset'] == 'GFDL-CM3':
                assert dataset['alias'] == 'CMIP6_GFDL-CM3'
            elif dataset['dataset'] == 'EC-EARTH':
                assert dataset['alias'] == 'CMIP6_EC-EARTH'
            else:
                assert dataset['alias'] == 'CMIP6_HADGEM'
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


@pytest.mark.parametrize('invalid_arg', ['crop', 'shapefile', 'method'])
def test_extract_shape_raises(tmp_path, patched_datafinder, config_user,
                              invalid_arg):
    content = dedent(f"""
        preprocessors:
          test:
            extract_shape:
              {invalid_arg}: x

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
    with pytest.raises(RecipeError) as exc:
        get_recipe(tmp_path, content, config_user)
        assert 'extract_shape' in exc.value
        assert invalid_arg in exc.value
