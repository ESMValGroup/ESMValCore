import os
import re
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from textwrap import dedent
from unittest.mock import create_autospec

import iris
import pytest
import yaml
from nested_lookup import get_occurrence_of_value
from PIL import Image

import esmvalcore
import esmvalcore._task
from esmvalcore._recipe.recipe import (
    _get_input_datasets,
    _representative_datasets,
    read_recipe_file,
)
from esmvalcore._task import DiagnosticTask
from esmvalcore.config import Session
from esmvalcore.config._config import TASKSEP
from esmvalcore.config._diagnostics import TAGS
from esmvalcore.dataset import Dataset
from esmvalcore.exceptions import RecipeError
from esmvalcore.local import _get_output_file
from esmvalcore.preprocessor import DEFAULT_ORDER, PreprocessingTask
from tests.integration.test_provenance import check_provenance

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
    'frequency',
    'institute',
    'long_name',
    'mip',
    'modeling_realm',
    'preprocessor',
    'project',
    'short_name',
    'standard_name',
    'timerange',
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
    'remove_supplementary_variables',
    'save',
)

INITIALIZATION_ERROR_MSG = 'Could not create all tasks'


def create_test_file(filename, tracking_id=None):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    attributes = {}
    if tracking_id is not None:
        attributes['tracking_id'] = tracking_id
    cube = iris.cube.Cube([])
    cube.attributes.globals = attributes

    iris.save(cube, filename)


def _get_default_settings_for_chl(save_filename):
    """Get default preprocessor settings for chl."""
    defaults = {
        'remove_supplementary_variables': {},
        'save': {
            'compress': False,
            'filename': save_filename,
        }
    }
    return defaults


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

    monkeypatch.setattr(
        esmvalcore._recipe.to_datasets,
        'get_required',
        get_required,
    )


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


def get_recipe(tempdir: Path, content: str, session: Session):
    """Save and load recipe content."""
    recipe_file = tempdir / 'recipe_test.yml'
    # Add mandatory documentation section
    content = str(DEFAULT_DOCUMENTATION + content)
    recipe_file.write_text(content)

    recipe = read_recipe_file(recipe_file, session)

    return recipe


def test_recipe_missing_scripts(tmp_path, session):
    content = dedent("""
        datasets:
          - dataset: bcc-csm1-1

        diagnostics:
          diagnostic_name:
            variables:
              ta:
                project: CMIP5
                mip: Amon
                exp: historical
                ensemble: r1i1p1
                timerange: 1999/2002
        """)
    exc_message = ("Missing scripts section in diagnostic 'diagnostic_name'.")
    with pytest.raises(RecipeError) as exc:
        get_recipe(tmp_path, content, session)
    assert str(exc.value) == exc_message


def test_recipe_duplicate_var_script_name(tmp_path, session):
    content = dedent("""
        datasets:
          - dataset: bcc-csm1-1

        diagnostics:
          diagnostic_name:
            variables:
              ta:
                project: CMIP5
                mip: Amon
                exp: historical
                ensemble: r1i1p1
                start_year: 1999
                end_year: 2002
            scripts:
              ta:
                script: tmp_path / 'diagnostic.py'
        """)
    exc_message = ("Invalid script name 'ta' encountered in diagnostic "
                   "'diagnostic_name': scripts cannot have the same "
                   "name as variables.")
    with pytest.raises(RecipeError) as exc:
        get_recipe(tmp_path, content, session)
    assert str(exc.value) == exc_message


def test_recipe_no_script(tmp_path, session):
    content = dedent("""
        datasets:
          - dataset: bcc-csm1-1

        diagnostics:
          diagnostic_name:
            variables:
              ta:
                project: CMIP5
                mip: Amon
                exp: historical
                ensemble: r1i1p1
                start_year: 1999
                end_year: 2002
            scripts:
              script_name:
                argument: 1
        """)
    exc_message = ("No script defined for script 'script_name' in "
                   "diagnostic 'diagnostic_name'.")
    with pytest.raises(RecipeError) as exc:
        get_recipe(tmp_path, content, session)
    assert str(exc.value) == exc_message


def test_recipe_no_datasets(tmp_path, session):
    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              ta:
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
                   "'ta' in diagnostic 'diagnostic_name'.")
    with pytest.raises(RecipeError) as exc:
        get_recipe(tmp_path, content, session)
    assert str(exc.value) == exc_message


def test_recipe_duplicated_datasets(tmp_path, session):
    content = dedent("""
        datasets:
          - dataset: bcc-csm1-1
          - dataset: bcc-csm1-1

        diagnostics:
          diagnostic_name:
            variables:
              ta:
                project: CMIP5
                mip: Amon
                exp: historical
                ensemble: r1i1p1
                timerange: 1999/2002
            scripts: null
        """)
    exc_message = ("Duplicate dataset\n{'dataset': 'bcc-csm1-1'}\n"
                   "for variable 'ta' in diagnostic 'diagnostic_name'.")
    with pytest.raises(RecipeError) as exc:
        get_recipe(tmp_path, content, session)
    assert str(exc.value) == exc_message


def test_recipe_var_missing_args(tmp_path, session):
    content = dedent("""
        datasets:
          - dataset: bcc-csm1-1

        diagnostics:
          diagnostic_name:
            variables:
              ta:
                project: CMIP5
                exp: historical
                ensemble: r1i1p1
                timerange: 1999/2002
            scripts: null
        """)
    exc_message = ("Missing keys {'mip'} in\n{'dataset': 'bcc-csm1-1',"
                   "\n 'ensemble': 'r1i1p1',\n 'exp': 'historical',\n"
                   " 'project': 'CMIP5',\n 'short_name': 'ta',\n "
                   "'timerange': '1999/2002'}\nfor variable 'ta' "
                   "in diagnostic 'diagnostic_name'.")
    with pytest.raises(RecipeError) as exc:
        get_recipe(tmp_path, content, session)
    assert str(exc.value) == exc_message


@pytest.mark.parametrize('skip_nonexistent', [True, False])
def test_recipe_no_data(tmp_path, session, skip_nonexistent):
    content = dedent("""
        datasets:
          - dataset: GFDL-ESM2G

        diagnostics:
          diagnostic_name:
            variables:
              ta:
                project: CMIP5
                mip: Amon
                exp: historical
                ensemble: r1i1p1
                start_year: 1999
                end_year: 2002
            scripts: null
        """)
    session['skip_nonexistent'] = skip_nonexistent
    with pytest.raises(RecipeError) as error:
        get_recipe(tmp_path, content, session)
    if skip_nonexistent:
        msg = ("Did not find any input data for task diagnostic_name/ta")
    else:
        msg = ("Missing data for preprocessor diagnostic_name/ta:\n"
               "- Missing data for Dataset: .*")
    assert re.match(msg, error.value.failed_tasks[0].message)


@pytest.mark.parametrize('script_file', ['diagnostic.py', 'diagnostic.ncl'])
def test_simple_recipe(
    tmp_path,
    patched_datafinder,
    session,
    script_file,
    monkeypatch,
):

    def ncl_version():
        return '6.5'

    monkeypatch.setattr(esmvalcore._recipe.check, 'ncl_version', ncl_version)

    def which(interpreter):
        return interpreter

    monkeypatch.setattr(esmvalcore._task, 'which', which)

    script = tmp_path / script_file
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
                timerange: 1999/2002
                additional_datasets:
                  - dataset: MPI-ESM-LR
            scripts:
              script_name:
                script: {}
                custom_setting: 1
        """.format(script))

    recipe = get_recipe(tmp_path, content, session)
    # Check that datasets have been read and updated
    assert len(recipe.datasets) == 3
    for dataset in recipe.datasets:
        for key in MANDATORY_DATASET_KEYS:
            assert key in dataset.facets and dataset.facets[key]

    # Check that the correct tasks have been created
    datasets = recipe.datasets
    tasks = {t for task in recipe.tasks for t in task.flatten()}
    preproc_tasks = {t for t in tasks if isinstance(t, PreprocessingTask)}
    diagnostic_tasks = {t for t in tasks if isinstance(t, DiagnosticTask)}

    assert len(preproc_tasks) == 1
    for task in preproc_tasks:
        print("Task", task.name)
        assert task.order == list(DEFAULT_ORDER)
        for product in task.products:
            dataset = [
                d for d in datasets if _get_output_file(
                    d.facets, session.preproc_dir) == product.filename
            ][0]
            assert product.datasets == [dataset]
            attributes = dict(dataset.facets)
            attributes['filename'] = product.filename
            attributes['start_year'] = 1999
            attributes['end_year'] = 2002
            assert product.attributes == attributes
            for step in DEFAULT_PREPROCESSOR_STEPS:
                assert step in product.settings
            assert len(dataset.files) == 2

    assert len(diagnostic_tasks) == 1
    for task in diagnostic_tasks:
        print("Task", task.name)
        assert task.ancestors == list(preproc_tasks)
        assert task.script == str(script)
        for key in MANDATORY_SCRIPT_SETTINGS_KEYS:
            assert key in task.settings and task.settings[key]
        assert task.settings['custom_setting'] == 1

    # Check that NCL interface is enabled for NCL scripts.
    write_ncl_interface = script.suffix == '.ncl'
    assert datasets[0].session['write_ncl_interface'] == write_ncl_interface


def test_write_filled_recipe(tmp_path, patched_datafinder, session):
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
                timerange: '*'
                additional_datasets:
                  - dataset: MPI-ESM-LR
                    timerange: '*/P2Y'
            scripts:
              script_name:
                script: {}
                custom_setting: 1
        """.format(script))

    recipe = get_recipe(tmp_path, content, session)

    session.run_dir.mkdir(parents=True)
    esmvalcore._recipe.recipe.Recipe.write_filled_recipe(recipe)

    recipe_file = session.run_dir / 'recipe_test_filled.yml'
    assert recipe_file.is_file()

    updated_recipe_object = read_recipe_file(recipe_file, session)
    updated_recipe = updated_recipe_object._raw_recipe
    print(pformat(updated_recipe))
    assert get_occurrence_of_value(updated_recipe, value='*') == 0
    assert get_occurrence_of_value(updated_recipe, value='1990/2019') == 2
    assert get_occurrence_of_value(updated_recipe, value='1990/P2Y') == 1
    assert len(updated_recipe_object.datasets) == 3


def test_fx_preproc_error(tmp_path, patched_datafinder, session):
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
        get_recipe(tmp_path, content, session)
    assert str(rec_err_exp.value) == INITIALIZATION_ERROR_MSG
    assert str(rec_err_exp.value.failed_tasks[0].message) == msg


def test_default_preprocessor(tmp_path, patched_datafinder, session):
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

    recipe = get_recipe(tmp_path, content, session)

    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert len(task.products) == 1
    product = task.products.pop()
    preproc_dir = os.path.dirname(product.filename)
    assert preproc_dir.startswith(str(tmp_path))

    defaults = _get_default_settings_for_chl(product.filename)
    assert product.settings == defaults


def test_default_preprocessor_custom_order(tmp_path, patched_datafinder,
                                           session):
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

    recipe = get_recipe(tmp_path, content, session)

    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert len(task.products) == 1
    product = task.products.pop()
    preproc_dir = os.path.dirname(product.filename)
    assert preproc_dir.startswith(str(tmp_path))

    defaults = _get_default_settings_for_chl(product.filename)
    assert product.settings == defaults


def test_invalid_preprocessor(tmp_path, patched_datafinder, session):
    """Test the error message when the named prepreprocesor is not defined."""
    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              chl:
                preprocessor: not_defined
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

    with pytest.raises(RecipeError) as error:
        get_recipe(tmp_path, content, session)
    msg = "Unknown preprocessor 'not_defined' in .*"
    assert re.match(msg, error.value.failed_tasks[0].message)


def test_disable_preprocessor_function(tmp_path, patched_datafinder, session):
    """Test if default settings are used when ``custom_order`` is ``True``."""

    content = dedent("""
        datasets:
          - dataset: HadGEM3-GC31-LL
            ensemble: r1i1p1f1
            exp: historical
            grid: gn

        preprocessors:
          keep_supplementaries:
            remove_supplementary_variables: False

        diagnostics:
          diagnostic_name:
            variables:
              tas:
                preprocessor: keep_supplementaries
                project: CMIP6
                mip: Amon
                timerange: 2000/2005
                supplementaries:
                  - short_name: areacella
                    mip: fx
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, session)

    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert len(task.products) == 1
    product = task.products.pop()
    assert 'remove_supplementary_variables' not in product.settings


def test_default_fx_preprocessor(tmp_path, patched_datafinder, session):
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

    recipe = get_recipe(tmp_path, content, session)

    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert len(task.products) == 1
    product = task.products.pop()
    preproc_dir = os.path.dirname(product.filename)
    assert preproc_dir.startswith(str(tmp_path))

    defaults = {
        'remove_supplementary_variables': {},
        'save': {
            'compress': False,
            'filename': product.filename,
        }
    }
    assert product.settings == defaults


def test_empty_variable(tmp_path, patched_datafinder, session):
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

    recipe = get_recipe(tmp_path, content, session)
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert len(task.products) == 1
    product = task.products.pop()
    assert product.attributes['short_name'] == 'pr'
    assert product.attributes['dataset'] == 'CanESM2'


TEST_ISO_TIMERANGE = [
    ('*', '1990-2019'),
    ('1990/1992', '1990-1992'),
    ('19900101/19920101', '19900101-19920101'),
    ('19900101T12H00M00S/19920101T12H00M00',
     '19900101T12H00M00S-19920101T12H00M00'),
    ('1990/*', '1990-2019'),
    ('*/1992', '1990-1992'),
    ('1990/P2Y', '1990-P2Y'),
    ('19900101/P2Y2M1D', '19900101-P2Y2M1D'),
    ('19900101TH00M00S/P2Y2M1DT12H00M00S',
     '19900101TH00M00S-P2Y2M1DT12H00M00S'),
    ('P2Y/1992', 'P2Y-1992'),
    ('P1Y2M1D/19920101', 'P1Y2M1D-19920101'),
    ('P1Y2M1D/19920101T12H00M00S', 'P1Y2M1D-19920101T12H00M00S'),
    ('P2Y/*', 'P2Y-2019'),
    ('P2Y2M1D/*', 'P2Y2M1D-2019'),
    ('P2Y21DT12H00M00S/*', 'P2Y21DT12H00M00S-2019'),
    ('*/P2Y', '1990-P2Y'),
    ('*/P2Y2M1D', '1990-P2Y2M1D'),
    ('*/P2Y21DT12H00M00S', '1990-P2Y21DT12H00M00S'),
]


@pytest.mark.parametrize('input_time,output_time', TEST_ISO_TIMERANGE)
def test_recipe_iso_timerange(tmp_path, patched_datafinder, session,
                              input_time, output_time):
    """Test recipe with timerange tag."""
    content = dedent(f"""
        diagnostics:
          test:
            additional_datasets:
              - dataset: HadGEM3-GC31-LL
                project: CMIP6
                exp: historical
                ensemble: r2i1p1f1
                grid: gn
            variables:
              pr:
                mip: 3hr
                timerange: '{input_time}'
              areacella:
                mip: fx
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, session)
    assert len(recipe.tasks) == 2
    pr_task = [t for t in recipe.tasks if t.name.endswith('pr')][0]
    assert len(pr_task.products) == 1
    pr_product = pr_task.products.pop()

    filename = ('CMIP6_HadGEM3-GC31-LL_3hr_historical_r2i1p1f1_'
                f'pr_gn_{output_time}.nc')
    assert pr_product.filename.name == filename

    areacella_task = [t for t in recipe.tasks
                      if t.name.endswith('areacella')][0]
    assert len(areacella_task.products) == 1
    areacella_product = areacella_task.products.pop()

    filename = 'CMIP6_HadGEM3-GC31-LL_fx_historical_r2i1p1f1_areacella_gn.nc'
    assert areacella_product.filename.name == filename


@pytest.mark.parametrize('input_time,output_time', TEST_ISO_TIMERANGE)
def test_recipe_iso_timerange_as_dataset(tmp_path, patched_datafinder, session,
                                         input_time, output_time):
    """Test recipe with timerange tag in the datasets section."""
    content = dedent(f"""
        datasets:
          - dataset: HadGEM3-GC31-LL
            project: CMIP6
            exp: historical
            ensemble: r2i1p1f1
            grid: gn
            timerange: '{input_time}'
        diagnostics:
          test:
            variables:
              pr:
                mip: 3hr
                supplementary_variables:
                  - short_name: areacella
                    mip: fx
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, session)

    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert len(task.products) == 1
    product = task.products.pop()
    filename = ('CMIP6_HadGEM3-GC31-LL_3hr_historical_r2i1p1f1_'
                f'pr_gn_{output_time}.nc')
    assert product.filename.name == filename

    assert len(product.datasets) == 1
    dataset = product.datasets[0]
    assert len(dataset.supplementaries) == 1
    supplementary_ds = dataset.supplementaries[0]
    assert supplementary_ds.facets['short_name'] == 'areacella'
    assert 'timerange' not in supplementary_ds.facets


def test_reference_dataset(tmp_path, patched_datafinder, session, monkeypatch):
    levels = [100]
    get_reference_levels = create_autospec(
        esmvalcore._recipe.recipe.get_reference_levels, return_value=levels)
    monkeypatch.setattr(esmvalcore._recipe.recipe, 'get_reference_levels',
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
                  - dataset: GFDL-CM3
                  - dataset: MPI-ESM-LR
                reference_dataset: MPI-ESM-LR
              ch4:
                <<: *var
                preprocessor: test_from_cmor_table
                additional_datasets:
                  - dataset: GFDL-CM3

            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, session)

    assert len(recipe.tasks) == 2

    # Check that the reference dataset has been used
    task = next(t for t in recipe.tasks
                if t.name == 'diagnostic_name' + TASKSEP + 'ta')
    assert len(task.products) == 2
    product = next(p for p in task.products
                   if p.attributes['dataset'] == 'GFDL-CM3')
    reference = next(p for p in task.products
                     if p.attributes['dataset'] == 'MPI-ESM-LR')

    assert product.settings['regrid']['target_grid'] == reference.datasets[0]
    assert product.settings['extract_levels']['levels'] == levels

    get_reference_levels.assert_called_once_with(reference.datasets[0])

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


def test_reference_dataset_undefined(tmp_path, monkeypatch, session):
    content = dedent("""
        preprocessors:
          test_from_reference:
            extract_levels:
              levels: reference_dataset
              scheme: linear

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
                  - dataset: GFDL-CM3
                  - dataset: MPI-ESM-LR

            scripts: null
        """)
    with pytest.raises(RecipeError) as error:
        get_recipe(tmp_path, content, session)
    msg = ("Preprocessor 'test_from_reference' uses 'reference_dataset', but "
           "'reference_dataset' is not defined")
    assert msg in error.value.failed_tasks[0].message


def test_custom_preproc_order(tmp_path, patched_datafinder, session):
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

    recipe = get_recipe(tmp_path, content, session)

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
        else:
            assert False, f"invalid task {task.name}"


def test_derive(tmp_path, patched_datafinder, session):
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

    recipe = get_recipe(tmp_path, content, session)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'toz'

    # Check product content of tasks
    assert len(task.products) == 1
    product = task.products.pop()
    assert 'derive' in product.settings
    assert product.attributes['short_name'] == 'toz'

    assert len(product.datasets) == 2
    input_variables = {d.facets['short_name'] for d in product.datasets}
    assert input_variables == {'ps', 'tro3'}


def test_derive_not_needed(tmp_path, patched_datafinder, session):
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

    recipe = get_recipe(tmp_path, content, session)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name/toz'

    # Check product content of tasks
    assert len(task.products) == 1
    product = task.products.pop()
    assert 'derive' not in product.settings

    # Check dataset
    assert len(product.datasets) == 1
    dataset = product.datasets[0]
    assert dataset.facets['short_name'] == 'toz'
    assert dataset.files


def test_derive_with_fx_ohc(tmp_path, patched_datafinder, session):
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
    recipe = get_recipe(tmp_path, content, session)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'ohc'

    # Check products
    assert len(task.products) == 3
    for product in task.products:
        assert 'derive' in product.settings
        assert product.attributes['short_name'] == 'ohc'

        # Check datasets
        assert len(product.datasets) == 2
        thetao_ds = next(d for d in product.datasets
                         if d.facets['short_name'] == 'thetao')
        assert thetao_ds.facets['mip'] == 'Omon'
        volcello_ds = next(d for d in product.datasets
                           if d.facets['short_name'] == 'volcello')
        if volcello_ds.facets['project'] == 'CMIP6':
            mip = 'Ofx'
        else:
            mip = 'fx'
        assert volcello_ds.facets['mip'] == mip


def test_derive_with_fx_ohc_fail(tmp_path, patched_failing_datafinder,
                                 session):
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
        get_recipe(tmp_path, content, session)


def test_derive_with_optional_var(tmp_path, patched_datafinder,
                                  patched_tas_derivation, session):
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
    recipe = get_recipe(tmp_path, content, session)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'tas'

    # Check products
    assert len(task.products) == 3
    for product in task.products:
        assert 'derive' in product.settings
        assert product.attributes['short_name'] == 'tas'
        assert len(product.datasets) == 2
        pr_ds = next(d for d in product.datasets
                     if d.facets['short_name'] == 'pr')
        assert pr_ds.facets['mip'] == 'Amon'
        assert pr_ds.facets['timerange'] == '2000/2005'
        areacella_ds = next(d for d in product.datasets
                            if d.facets['short_name'] == 'areacella')
        assert areacella_ds.facets['mip'] == 'fx'
        assert 'timerange' not in areacella_ds.facets


def test_derive_with_optional_var_nodata(tmp_path, patched_failing_datafinder,
                                         patched_tas_derivation, session):
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
    recipe = get_recipe(tmp_path, content, session)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'tas'

    # Check products
    assert len(task.products) == 3
    for product in task.products:
        assert 'derive' in product.settings
        assert product.attributes['short_name'] == 'tas'

        # Check datasets
        assert len(product.datasets) == 1
        assert product.datasets[0].facets['short_name'] == 'pr'


def test_derive_contains_start_end_year(tmp_path, patched_datafinder, session):
    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              toz:
                project: CMIP5
                mip: Amon
                exp: historical
                timerange: '2000/2005'
                derive: true
                force_derivation: true
                additional_datasets:
                  - {dataset: GFDL-CM3,  ensemble: r1i1p1}
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, session)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()

    # Check that start_year and end_year are present in attributes
    assert len(task.products) == 1
    product = task.products.pop()
    assert 'derive' in product.settings
    assert product.attributes['short_name'] == 'toz'
    assert product.attributes['timerange'] == '2000/2005'
    assert product.attributes['start_year'] == 2000
    assert product.attributes['end_year'] == 2005


@pytest.mark.parametrize('force_derivation', [True, False])
def test_derive_timerange_wildcard(tmp_path, patched_datafinder, session,
                                   force_derivation):

    content = dedent(f"""
        diagnostics:
          diagnostic_name:
            variables:
              toz:
                project: CMIP5
                mip: Amon
                exp: historical
                timerange: '*'
                derive: true
                force_derivation: {force_derivation}
                additional_datasets:
                  - dataset: GFDL-CM3
                    ensemble: r1i1p1
            scripts: null
        """)

    recipe = get_recipe(tmp_path, content, session)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()

    # Check that start_year and end_year are present in attributes
    assert len(task.products) == 1
    product = task.products.pop()
    if force_derivation:
        assert 'derive' in product.settings
    assert product.attributes['short_name'] == 'toz'
    assert product.attributes['timerange'] == '1990/2019'
    assert product.attributes['start_year'] == 1990
    assert product.attributes['end_year'] == 2019


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


def simulate_preprocessor_run(task):
    """Simulate preprocessor run."""
    task._initialize_product_provenance()
    for product in task.products:
        create_test_file(product.filename)
        product.save_provenance()


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
    with open(provenance, 'w', encoding='utf-8') as file:
        yaml.safe_dump({diagnostic_file: record, plot_file: record}, file)

    diagnostic_task._collect_provenance()
    return record


def test_diagnostic_task_provenance(
    tmp_path,
    patched_datafinder,
    session,
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

    recipe = get_recipe(tmp_path, content, session)
    preproc_task = next(t for t in recipe.tasks.flatten()
                        if isinstance(t, PreprocessingTask))
    simulate_preprocessor_run(preproc_task)

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

    # Test that provenance was saved to xml and info embedded in netcdf
    product = next(
        iter(p for p in diagnostic_task.products
             if p.filename.endswith('.nc')))
    cube = iris.load_cube(product.filename)
    assert cube.attributes['software'].startswith("Created with ESMValTool v")
    assert cube.attributes['caption'] == record['caption']
    prefix = os.path.splitext(product.filename)[0] + '_provenance'
    assert os.path.exists(prefix + '.xml')


def test_alias_generation(tmp_path, patched_datafinder, session):
    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              pr:
                project: CMIP5
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2005
                grid: gn
                type: reanaly
                tier: 2
                version: latest
                domain: EUR-11
                rcm_version: 1
                additional_datasets:
                  - {dataset: GFDL-CM3,  ensemble: r1i1p1}
                  - {dataset: EC-EARTH,  ensemble: r1i1p1}
                  - {dataset: EC-EARTH,  ensemble: r2i1p1}
                  - {dataset: EC-EARTH,  ensemble: r3i1p1, alias: my_alias}
                  - {dataset: FGOALS-g3, sub_experiment: s1960, ensemble: r1, institute: CAS}
                  - {dataset: FGOALS-g3, sub_experiment: s1961, ensemble: r1, institute: CAS}
                  - {project: OBS, dataset: ERA-Interim,  version: 1}
                  - {project: OBS, dataset: ERA-Interim,  version: 2}
                  - {project: CMIP6, activity: CMP, dataset: GF3, ensemble: r1, institute: fake}
                  - {project: CMIP6, activity: CMP, dataset: GF2, ensemble: r1, institute: fake}
                  - {project: CMIP6, activity: HRMP, dataset: EC, ensemble: r1, institute: fake}
                  - {project: CMIP6, activity: HRMP, dataset: HA, ensemble: r1, institute: fake}
                  - {project: CORDEX, driver: ICHEC-EC-EARTH, dataset: RCA4, ensemble: r1, mip: mon, institute: SMHI}
                  - {project: CORDEX, driver: MIROC-MIROC5, dataset: RCA4, ensemble: r1, mip: mon, institute: SMHI}
            scripts: null
        """)  # noqa:

    recipe = get_recipe(tmp_path, content, session)
    assert len(recipe.datasets) == 14
    for dataset in recipe.datasets:
        if dataset['project'] == 'CMIP5':
            if dataset['dataset'] == 'GFDL-CM3':
                assert dataset['alias'] == 'CMIP5_GFDL-CM3'
            elif dataset['dataset'] == 'FGOALS-g3':
                if dataset['sub_experiment'] == 's1960':
                    assert dataset['alias'] == 'CMIP5_FGOALS-g3_s1960'
                else:
                    assert dataset['alias'] == 'CMIP5_FGOALS-g3_s1961'
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
        elif dataset['project'] == 'CORDEX':
            if dataset['driver'] == 'ICHEC-EC-EARTH':
                assert dataset['alias'] == 'CORDEX_ICHEC-EC-EARTH'
            else:
                assert dataset['alias'] == 'CORDEX_MIROC-MIROC5'
        else:
            if dataset['version'] == 1:
                assert dataset['alias'] == 'OBS_1'
            else:
                assert dataset['alias'] == 'OBS_2'


def test_concatenation(tmp_path, patched_datafinder, session):
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

    recipe = get_recipe(tmp_path, content, session)
    assert len(recipe.datasets) == 2
    for dataset in recipe.datasets:
        if dataset['exp'] == 'historical':
            assert dataset['alias'] == 'historical'
        else:
            assert dataset['alias'] == 'historical-rcp85'


def test_ensemble_expansion(tmp_path, patched_datafinder, session):
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

    recipe = get_recipe(tmp_path, content, session)
    assert len(recipe.datasets) == 3
    assert recipe.datasets[0]['ensemble'] == 'r1i1p1'
    assert recipe.datasets[1]['ensemble'] == 'r2i1p1'
    assert recipe.datasets[2]['ensemble'] == 'r3i1p1'


def test_extract_shape(tmp_path, patched_datafinder, session):
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
    shapefile = session['auxiliary_data_dir'] / Path('test.shp')
    shapefile.parent.mkdir(parents=True, exist_ok=True)
    shapefile.touch()

    recipe = get_recipe(tmp_path, content, session)

    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert len(task.products) == 1
    product = task.products.pop()
    assert product.settings['extract_shape']['shapefile'] == shapefile


@pytest.mark.parametrize('invalid_arg',
                         ['shapefile', 'method', 'crop', 'decomposed'])
def test_extract_shape_raises(tmp_path, patched_datafinder, session,
                              invalid_arg):
    TAGS.set_tag_values(TAGS_FOR_TESTING)

    # Create shapefile
    shapefile = session['auxiliary_data_dir'] / Path('test.shp')
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
                  - dataset: GFDL-CM3
            scripts: null
        """)

    # Add invalid argument
    recipe = yaml.safe_load(content)
    recipe['preprocessors']['test']['extract_shape'][invalid_arg] = 'x'
    content = yaml.safe_dump(recipe)

    with pytest.raises(RecipeError) as exc:
        get_recipe(tmp_path, content, session)

    assert str(exc.value) == INITIALIZATION_ERROR_MSG
    assert 'extract_shape' in exc.value.failed_tasks[0].message
    assert invalid_arg in exc.value.failed_tasks[0].message


def _test_output_product_consistency(products, preprocessor, statistics):
    product_out = defaultdict(list)

    for i, product in enumerate(products):
        settings = product.settings.get(preprocessor)
        if settings:
            output_products = settings['output_products']

            for identifier, statistic_out in output_products.items():
                for statistic, preproc_file in statistic_out.items():
                    product_out[identifier, statistic].append(preproc_file)

    # Make sure that output products are consistent
    for (identifier, statistic), value in product_out.items():
        assert statistic in statistics
        assert len(set(value)) == 1, 'Output products are not equal'

    return product_out


def test_ensemble_statistics(tmp_path, patched_datafinder, session):
    statistics = ['mean', 'max']
    diagnostic = 'diagnostic_name'
    variable = 'pr'
    preprocessor = 'ensemble_statistics'

    content = dedent(f"""
         preprocessors:
           default: &default
             custom_order: true
             area_statistics:
               operator: mean
             {preprocessor}:
               statistics: {statistics}

         diagnostics:
           {diagnostic}:
             variables:
               {variable}:
                 project: CMIP5
                 mip: Amon
                 start_year: 2000
                 end_year: 2002
                 preprocessor: default
                 additional_datasets:
                   - {{dataset: CanESM2, exp: [historical, rcp45],
                     ensemble: "r(1:2)i1p1"}}
                   - {{dataset: CCSM4, exp: [historical, rcp45],
                     ensemble: "r(1:2)i1p1"}}
             scripts: null
    """)

    recipe = get_recipe(tmp_path, content, session)
    datasets = set([ds['dataset'] for ds in recipe.datasets])
    task = next(iter(recipe.tasks))

    products = task.products
    product_out = _test_output_product_consistency(products, preprocessor,
                                                   statistics)

    assert len(product_out) == len(datasets) * len(statistics)

    task._initialize_product_provenance()
    assert next(iter(products)).provenance is not None


def test_multi_model_statistics(tmp_path, patched_datafinder, session):
    statistics = ['mean', 'max']
    diagnostic = 'diagnostic_name'
    variable = 'pr'
    preprocessor = 'multi_model_statistics'

    content = dedent(f"""
        preprocessors:
          default: &default
            custom_order: true
            area_statistics:
              operator: mean
            {preprocessor}:
              span: overlap
              statistics: {statistics}

        diagnostics:
          {diagnostic}:
            variables:
              {variable}:
                project: CMIP5
                mip: Amon
                start_year: 2000
                end_year: 2002
                preprocessor: default
                additional_datasets:
                  - {{dataset: CanESM2, exp: [historical, rcp45],
                    ensemble: "r(1:2)i1p1"}}
                  - {{dataset: CCSM4, exp: [historical, rcp45],
                    ensemble: "r(1:2)i1p1"}}
            scripts: null
    """)

    recipe = get_recipe(tmp_path, content, session)
    task = next(iter(recipe.tasks))

    products = task.products
    product_out = _test_output_product_consistency(products, preprocessor,
                                                   statistics)

    assert len(product_out) == len(statistics)

    task._initialize_product_provenance()
    assert next(iter(products)).provenance is not None


def test_multi_model_statistics_exclude(tmp_path, patched_datafinder, session):
    statistics = ['mean', 'max']
    diagnostic = 'diagnostic_name'
    variable = 'pr'
    preprocessor = 'multi_model_statistics'

    content = dedent(f"""
        preprocessors:
          default: &default
            custom_order: true
            area_statistics:
              operator: mean
            {preprocessor}:
              span: overlap
              statistics: {statistics}
              groupby: ['project']
              exclude: ['TEST']

        diagnostics:
          {diagnostic}:
            variables:
              {variable}:
                project: CMIP5
                mip: Amon
                start_year: 2000
                end_year: 2002
                preprocessor: default
                additional_datasets:
                  - {{dataset: CanESM2, exp: [historical, rcp45],
                    ensemble: "r(1:2)i1p1"}}
                  - {{dataset: CCSM4, exp: [historical, rcp45],
                    ensemble: "r(1:2)i1p1"}}
                  - {{dataset: TEST, project: OBS, type: reanaly, version: 1,
                     tier: 1}}
            scripts: null
    """)

    recipe = get_recipe(tmp_path, content, session)
    task = next(iter(recipe.tasks))

    products = task.products
    product_out = _test_output_product_consistency(products, preprocessor,
                                                   statistics)

    assert len(product_out) == len(statistics)
    assert 'OBS' not in product_out
    for id, prods in product_out:
        assert id != 'OBS'
        assert id == 'CMIP5'
    task._initialize_product_provenance()
    assert next(iter(products)).provenance is not None


def test_groupby_combined_statistics(tmp_path, patched_datafinder, session):
    diagnostic = 'diagnostic_name'
    variable = 'pr'

    mm_statistics = ['mean', 'max']
    mm_preprocessor = 'multi_model_statistics'
    ens_statistics = ['mean', 'median']
    ens_preprocessor = 'ensemble_statistics'

    groupby = [ens_preprocessor, 'tag']

    content = dedent(f"""
        preprocessors:
          default: &default
            custom_order: true
            area_statistics:
              operator: mean
            {ens_preprocessor}:
              span: 'overlap'
              statistics: {ens_statistics}
            {mm_preprocessor}:
              span: overlap
              groupby: {groupby}
              statistics: {mm_statistics}

        diagnostics:
          {diagnostic}:
            variables:
              {variable}:
                project: CMIP5
                mip: Amon
                start_year: 2000
                end_year: 2002
                preprocessor: default
                additional_datasets:
                  - {{dataset: CanESM2, exp: [historical, rcp45],
                    ensemble: "r(1:2)i1p1", tag: group1}}
                  - {{dataset: CCSM4, exp: [historical, rcp45],
                    ensemble: "r(1:2)i1p1", tag: group2}}
            scripts: null
    """)

    recipe = get_recipe(tmp_path, content, session)
    datasets = set([ds['dataset'] for ds in recipe.datasets])

    products = next(iter(recipe.tasks)).products

    ens_products = _test_output_product_consistency(
        products,
        ens_preprocessor,
        ens_statistics,
    )

    mm_products = _test_output_product_consistency(
        products,
        mm_preprocessor,
        mm_statistics,
    )

    assert len(ens_products) == len(datasets) * len(ens_statistics)
    assert len(
        mm_products) == len(mm_statistics) * len(ens_statistics) * len(groupby)


def test_weighting_landsea_fraction(tmp_path, patched_datafinder, session):
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
                supplementary_variables:
                  - short_name: sftlf
                    mip: fx
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, session)

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
        assert len(product.datasets) == 1
        dataset = product.datasets[0]
        assert len(dataset.supplementaries) == 1
        assert dataset.supplementaries[0].facets['short_name'] == 'sftlf'


def test_weighting_landsea_fraction_no_fx(tmp_path, patched_failing_datafinder,
                                          session):
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

    with pytest.raises(RecipeError):
        get_recipe(tmp_path, content, session)


def test_weighting_landsea_fraction_exclude(tmp_path, patched_datafinder,
                                            session):
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
                  - {dataset: TEST, project: obs4MIPs,
                     supplementary_variables: [{short_name: sftlf, mip: fx}]}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, session)

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
                                                 session):
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
        get_recipe(tmp_path, content, session)
    assert str(exc_info.value) == INITIALIZATION_ERROR_MSG
    assert str(exc_info.value.failed_tasks[0].message) == (
        "Preprocessor 'landfrac_weighting' uses 'alternative_dataset', but "
        "'alternative_dataset' is not defined for variable 'gpp' of "
        "diagnostic 'diagnostic_name'.")


def test_area_statistics(tmp_path, patched_datafinder, session):
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
                supplementary_variables:
                  - short_name: areacella
                    mip: fx
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, session)

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
        assert len(product.datasets) == 1
        dataset = product.datasets[0]
        assert len(dataset.supplementaries) == 1
        assert dataset.supplementaries[0].facets['short_name'] == 'areacella'


def test_landmask(tmp_path, patched_datafinder, session):
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
                supplementary_variables:
                  - short_name: sftlf
                    mip: fx
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, session)

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
        assert len(product.datasets) == 1
        dataset = product.datasets[0]
        assert len(dataset.supplementaries) == 1
        assert dataset.supplementaries[0].facets['short_name'] == 'sftlf'


def test_landmask_no_fx(tmp_path, patched_failing_datafinder, session):
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
                  - {dataset: CanESM5, project: CMIP6, grid: gn,
                     ensemble: r1i1p1f1}
                  - {dataset: TEST, project: obs4MIPs, level: 1, version: 1,
                     tier: 1}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, session)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == 'diagnostic_name' + TASKSEP + 'gpp'

    # Check masking
    assert len(task.products) == 3
    for product in task.products:
        assert 'mask_landsea' in product.settings
        settings = product.settings['mask_landsea']
        assert len(settings) == 1
        assert settings['mask_out'] == 'sea'
        assert len(product.datasets) == 1
        dataset = product.datasets[0]
        assert dataset.supplementaries == []


def test_wrong_project(tmp_path, patched_datafinder, session):
    content = dedent("""
        preprocessors:
          preproc:
           volume_statistics:
             operator: mean
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
        get_recipe(tmp_path, content, session)
    assert str(wrong_proj.value) == msg


def test_multimodel_mask(tmp_path, patched_datafinder, session):
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
    recipe = get_recipe(tmp_path, content, session)

    # Check generated tasks
    assert len(recipe.tasks) == 1
    task = recipe.tasks.pop()
    assert task.name == f'diagnostic_name{TASKSEP}tas'

    # Check mask_multimodel
    assert len(task.products) == 3
    for product in task.products:
        assert 'mask_multimodel' in product.settings
        assert product.settings['mask_multimodel'] == {}


def test_obs4mips_case_correct(tmp_path, patched_datafinder, session):
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
    recipe = get_recipe(tmp_path, content, session)
    dataset = recipe.datasets[0]
    assert dataset['project'] == 'obs4MIPs'


def test_recipe_run(tmp_path, patched_datafinder, session, mocker):
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
    session['download_dir'] = tmp_path / 'download_dir'
    session['search_esgf'] = 'when_missing'

    mocker.patch.object(esmvalcore._recipe.recipe.esgf,
                        'download',
                        create_autospec=True)

    recipe = get_recipe(tmp_path, content, session)

    recipe.tasks.run = mocker.Mock()
    recipe.write_filled_recipe = mocker.Mock()
    recipe.write_html_summary = mocker.Mock()
    recipe.run()

    esmvalcore._recipe.recipe.esgf.download.assert_called_once_with(
        set(), session['download_dir'])
    recipe.tasks.run.assert_called_once_with(
        max_parallel_tasks=session['max_parallel_tasks'])
    recipe.write_filled_recipe.assert_called_once()
    recipe.write_html_summary.assert_called_once()


def test_representative_dataset_regular_var(patched_datafinder, session):
    """Test ``_representative_dataset`` with regular variable."""
    variable = {
        'dataset': 'ICON',
        'exp': 'atm_amip-rad_R2B4_r1i1p1f1',
        'frequency': 'mon',
        'mip': 'Amon',
        'original_short_name': 'tas',
        'project': 'ICON',
        'short_name': 'tas',
        'timerange': '1990/2000',
        'var_type': 'atm_2d_ml',
    }
    dataset = Dataset(**variable)
    dataset.session = session
    datasets = _representative_datasets(dataset)
    assert len(datasets) == 1
    filename = datasets[0].files[0]
    path = Path(filename)
    assert path.name == 'atm_amip-rad_R2B4_r1i1p1f1_atm_2d_ml_1990_1999.nc'


@pytest.mark.parametrize('force_derivation', [True, False])
def test_representative_dataset_derived_var(patched_datafinder, session,
                                            force_derivation):
    """Test ``_representative_dataset`` with derived variable."""
    variable = {
        'dataset': 'ICON',
        'derive': True,
        'exp': 'atm_amip-rad_R2B4_r1i1p1f1',
        'force_derivation': force_derivation,
        'frequency': 'mon',
        'mip': 'Amon',
        'original_short_name': 'alb',
        'project': 'ICON',
        'short_name': 'alb',
        'timerange': '1990/2000',
        'var_type': 'atm_2d_ml',
    }
    dataset = Dataset(**variable)
    dataset.session = session
    representative_datasets = _representative_datasets(dataset)

    expected_facets = {
        # Already present in variable
        'dataset': 'ICON',
        'derive': True,
        'exp': 'atm_amip-rad_R2B4_r1i1p1f1',
        'force_derivation': force_derivation,
        'frequency': 'mon',
        'mip': 'Amon',
        'project': 'ICON',
        'timerange': '1990/2000',
        # Added by _add_cmor_info
        'modeling_realm': ['atmos'],
        'units': 'W m-2',
        # Added by _add_extra_facets
        'var_type': 'atm_2d_ml',
    }
    if force_derivation:
        expected_datasets = [
            Dataset(
                short_name='rsdscs',
                long_name='Surface Downwelling Clear-Sky Shortwave Radiation',
                original_short_name='rsdscs',
                standard_name=(
                    'surface_downwelling_shortwave_flux_in_air_assuming_clear_'
                    'sky'
                ),
                **expected_facets,
            ),
            Dataset(
                short_name='rsuscs',
                long_name='Surface Upwelling Clear-Sky Shortwave Radiation',
                original_short_name='rsuscs',
                standard_name=(
                    'surface_upwelling_shortwave_flux_in_air_assuming_clear_'
                    'sky'
                ),
                **expected_facets,
            ),
        ]
    else:
        expected_datasets = [dataset]
    for dataset in expected_datasets:
        dataset.session = session

    assert representative_datasets == expected_datasets


def test_get_derive_input_variables(patched_datafinder, session):
    """Test ``_get_derive_input_variables``."""
    alb_facets = {
        'dataset': 'ICON',
        'derive': True,
        'exp': 'atm_amip-rad_R2B4_r1i1p1f1',
        'force_derivation': True,
        'frequency': 'mon',
        'mip': 'Amon',
        'original_short_name': 'alb',
        'project': 'ICON',
        'short_name': 'alb',
        'timerange': '1990/2000',
    }
    alb = Dataset(**alb_facets)
    alb.session = session

    rsdscs_facets = {
        # Added by get_required
        'short_name': 'rsdscs',
        # Already present in variables
        'dataset': 'ICON',
        'derive': True,
        'exp': 'atm_amip-rad_R2B4_r1i1p1f1',
        'force_derivation': True,
        'frequency': 'mon',
        'mip': 'Amon',
        'project': 'ICON',
        'timerange': '1990/2000',
        # Added by _add_cmor_info
        'standard_name':
        'surface_downwelling_shortwave_flux_in_air_assuming_clear_sky',
        'long_name': 'Surface Downwelling Clear-Sky Shortwave Radiation',
        'modeling_realm': ['atmos'],
        'original_short_name': 'rsdscs',
        'units': 'W m-2',
        # Added by _add_extra_facets
        'var_type': 'atm_2d_ml',
    }
    rsdscs = Dataset(**rsdscs_facets)
    rsdscs.session = session

    rsuscs_facets = {
        # Added by get_required
        'short_name': 'rsuscs',
        # Already present in variables
        'dataset': 'ICON',
        'derive': True,
        'exp': 'atm_amip-rad_R2B4_r1i1p1f1',
        'force_derivation': True,
        'frequency': 'mon',
        'mip': 'Amon',
        'project': 'ICON',
        'timerange': '1990/2000',
        # Added by _add_cmor_info
        'standard_name':
        'surface_upwelling_shortwave_flux_in_air_assuming_clear_sky',
        'long_name': 'Surface Upwelling Clear-Sky Shortwave Radiation',
        'modeling_realm': ['atmos'],
        'original_short_name': 'rsuscs',
        'units': 'W m-2',
        # Added by _add_extra_facets
        'var_type': 'atm_2d_ml',
    }
    rsuscs = Dataset(**rsuscs_facets)
    rsuscs.session = session

    alb_derive_input = _get_input_datasets(alb)
    assert alb_derive_input == [rsdscs, rsuscs]


TEST_DIAG_SELECTION = [
    (None, {'d1/tas', 'd1/s1', 'd2/s1', 'd3/s1', 'd3/s2', 'd4/s1'}),
    ({''}, set()),
    ({'wrong_diag/*'}, set()),
    ({'d1/*'}, {'d1/tas', 'd1/s1'}),
    ({'d2/*'}, {'d1/tas', 'd1/s1', 'd2/s1'}),
    ({'d3/*'}, {'d1/tas', 'd1/s1', 'd2/s1', 'd3/s1', 'd3/s2'}),
    ({'d4/*'}, {'d1/tas', 'd1/s1', 'd2/s1', 'd3/s2', 'd4/s1'}),
    ({'wrong_diag/*', 'd1/*'}, {'d1/tas', 'd1/s1'}),
    ({'d1/tas'}, {'d1/tas'}),
    ({'d1/tas', 'd2/*'}, {'d1/tas', 'd1/s1', 'd2/s1'}),
    ({'d1/tas', 'd3/s1'}, {'d1/tas', 'd3/s1', 'd1/s1'}),
    ({'d4/*',
      'd3/s1'}, {'d1/tas', 'd1/s1', 'd2/s1', 'd3/s1', 'd3/s2', 'd4/s1'}),
]


@pytest.mark.parametrize('diags_to_run,tasks_run', TEST_DIAG_SELECTION)
def test_diag_selection(tmp_path, patched_datafinder, session, diags_to_run,
                        tasks_run):
    """Test selection of individual diagnostics via --diagnostics option."""
    TAGS.set_tag_values(TAGS_FOR_TESTING)
    script = tmp_path / 'diagnostic.py'
    script.write_text('')

    if diags_to_run is not None:
        session['diagnostics'] = diags_to_run

    content = dedent("""
        diagnostics:

          d1:
            variables:
              tas:
                project: CMIP6
                mip: Amon
                exp: historical
                start_year: 2000
                end_year: 2000
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - dataset: CanESM5
            scripts:
              s1:
                script: {script}

          d2:
            scripts:
              s1:
                script: {script}
                ancestors: [d1/*]

          d3:
            scripts:
              s1:
                script: {script}
                ancestors: [d1/s1]
              s2:
                script: {script}
                ancestors: [d2/*]

          d4:
            scripts:
              s1:
                script: {script}
                ancestors: [d3/s2]
        """).format(script=script)

    recipe = get_recipe(tmp_path, content, session)
    task_names = {task.name for task in recipe.tasks.flatten()}

    assert tasks_run == task_names


@pytest.mark.parametrize(
    'preproc', ['multi_model_statistics', 'ensemble_statistics']
)
def test_mm_stats_invalid_arg(preproc, tmp_path, patched_datafinder, session):
    content = dedent(f"""
        preprocessors:
          test:
            {preproc}:
              span: overlap
              statistics: [mean]
              invalid_argument: 1

        diagnostics:
          diagnostic_name:
            variables:
              chl_default:
                short_name: chl
                mip: Oyr
                preprocessor: test
                timerange: '2000/2010'
                additional_datasets:
                  - project: CMIP5
                    dataset: CanESM2
                    exp: historical
                    ensemble: r1i1p1
            scripts: null
        """)
    with pytest.raises(ValueError):
        get_recipe(tmp_path, content, session)


@pytest.mark.parametrize(
    'preproc', ['multi_model_statistics', 'ensemble_statistics']
)
def test_mm_stats_missing_arg(preproc, tmp_path, patched_datafinder, session):
    content = dedent(f"""
        preprocessors:
          test:
            {preproc}:

        diagnostics:
          diagnostic_name:
            variables:
              chl_default:
                short_name: chl
                mip: Oyr
                preprocessor: test
                timerange: '2000/2010'
                additional_datasets:
                  - project: CMIP5
                    dataset: CanESM2
                    exp: historical
                    ensemble: r1i1p1
            scripts: null
        """)
    with pytest.raises(ValueError):
        get_recipe(tmp_path, content, session)


@pytest.mark.parametrize(
    'preproc', ['multi_model_statistics', 'ensemble_statistics']
)
def test_mm_stats_invalid_stats(
    preproc, tmp_path, patched_datafinder, session
):
    content = dedent(f"""
        preprocessors:
          test:
            {preproc}:
              span: overlap
              statistics:
                - operator_is_missing: here
                  but_we_need: it

        diagnostics:
          diagnostic_name:
            variables:
              chl_default:
                short_name: chl
                mip: Oyr
                preprocessor: test
                timerange: '2000/2010'
                additional_datasets:
                  - project: CMIP5
                    dataset: CanESM2
                    exp: historical
                    ensemble: r1i1p1
            scripts: null
        """)
    msg = (
        "`statistic` given as dictionary, but missing required key `operator`"
    )
    with pytest.raises(RecipeError) as rec_err_exp:
        get_recipe(tmp_path, content, session)
    assert str(rec_err_exp.value) == INITIALIZATION_ERROR_MSG
    assert msg in str(rec_err_exp.value.failed_tasks[0].message)


@pytest.mark.parametrize(
    'statistics',
    [
        {'invalid_value': 1},
        {'percent': 10, 'invalid_value': 1},
        {'percent': 10, 'weights': False},
    ]
)
@pytest.mark.parametrize(
    'preproc', ['multi_model_statistics', 'ensemble_statistics']
)
def test_mm_stats_invalid_stat_kwargs(
    preproc, statistics, tmp_path, patched_datafinder, session
):
    statistics['operator'] = 'wpercentile'
    content = dedent(f"""
        preprocessors:
          test:
            {preproc}:
              span: overlap
              statistics: [{statistics}]

        diagnostics:
          diagnostic_name:
            variables:
              chl_default:
                short_name: chl
                mip: Oyr
                preprocessor: test
                timerange: '2000/2010'
                additional_datasets:
                  - project: CMIP5
                    dataset: CanESM2
                    exp: historical
                    ensemble: r1i1p1
            scripts: null
        """)
    msg = (
        f"Invalid options for {preproc}: Invalid kwargs for operator "
        f"'wpercentile': "
    )
    with pytest.raises(RecipeError) as rec_err_exp:
        get_recipe(tmp_path, content, session)
    assert str(rec_err_exp.value) == INITIALIZATION_ERROR_MSG
    assert msg in str(rec_err_exp.value.failed_tasks[0].message)


@pytest.mark.parametrize(
    'preproc',
    [
        'area_statistics',
        'axis_statistics',
        'meridional_statistics',
        'volume_statistics',
        'zonal_statistics',
        'rolling_window_statistics',
    ]
)
def test_statistics_missing_operator_no_default_fail(
    preproc, tmp_path, patched_datafinder, session
):
    content = dedent(f"""
        preprocessors:
          test:
            {preproc}:

        diagnostics:
          diagnostic_name:
            variables:
              chl_default:
                short_name: chl
                mip: Oyr
                preprocessor: test
                timerange: '2000/2010'
                additional_datasets:
                  - project: CMIP5
                    dataset: CanESM2
                    exp: historical
                    ensemble: r1i1p1
            scripts: null
        """)
    msg = "Missing required argument"
    with pytest.raises(ValueError, match=msg):
        get_recipe(tmp_path, content, session)


@pytest.mark.parametrize(
    'preproc,option',
    [
        ('annual_statistics', ''),
        ('climate_statistics', ''),
        ('daily_statistics', ''),
        ('decadal_statistics', ''),
        ('hourly_statistics', 'hours: 1'),
        ('monthly_statistics', ''),
        ('seasonal_statistics', ''),
    ]
)
def test_statistics_missing_operator_with_default(
    preproc, option, tmp_path, patched_datafinder, session
):
    content = dedent(f"""
        preprocessors:
          test:
            {preproc}:
              {option}

        diagnostics:
          diagnostic_name:
            variables:
              chl_default:
                short_name: chl
                mip: Oyr
                preprocessor: test
                timerange: '2000/2010'
                additional_datasets:
                  - project: CMIP5
                    dataset: CanESM2
                    exp: historical
                    ensemble: r1i1p1
            scripts: null
        """)
    get_recipe(tmp_path, content, session)


@pytest.mark.parametrize(
    'preproc,preproc_kwargs',
    [
        ('annual_statistics', {'invalid_value': 1}),
        ('area_statistics', {'percent': 10, 'invalid_value': 1}),
        ('axis_statistics', {'percent': 10, 'weights': False}),
        ('climate_statistics', {'invalid_value': 1}),
        ('daily_statistics', {'percent': 10, 'invalid_value': 1}),
        ('decadal_statistics', {'percent': 10, 'weights': False}),
        ('hourly_statistics', {'invalid_value': 1, 'hours': 2}),
        ('meridional_statistics', {'percent': 10, 'invalid_value': 1}),
        ('monthly_statistics', {'percent': 10, 'weights': False}),
        ('seasonal_statistics', {'invalid_value': 1}),
        ('volume_statistics', {'percent': 10, 'weights': False}),
        ('zonal_statistics', {'invalid_value': 1}),
        ('rolling_window_statistics', {'percent': 10, 'invalid_value': 1}),
    ]
)
def test_statistics_invalid_kwargs(
    preproc, preproc_kwargs, tmp_path, patched_datafinder, session
):
    preproc_kwargs['operator'] = 'wpercentile'
    content = dedent(f"""
        preprocessors:
          test:
            {preproc}: {preproc_kwargs}

        diagnostics:
          diagnostic_name:
            variables:
              chl_default:
                short_name: chl
                mip: Oyr
                preprocessor: test
                timerange: '2000/2010'
                additional_datasets:
                  - project: CMIP5
                    dataset: CanESM2
                    exp: historical
                    ensemble: r1i1p1
            scripts: null
        """)
    msg = (
        f"Invalid options for {preproc}: Invalid kwargs for operator "
        f"'wpercentile': "
    )
    with pytest.raises(RecipeError) as rec_err_exp:
        get_recipe(tmp_path, content, session)
    assert str(rec_err_exp.value) == INITIALIZATION_ERROR_MSG
    assert msg in str(rec_err_exp.value.failed_tasks[0].message)


def test_hourly_statistics(tmp_path, patched_datafinder, session):
    content = dedent("""
        preprocessors:
          test:
            hourly_statistics:
              hours: 10
              operator: percentile
              percent: 50

        diagnostics:
          diagnostic_name:
            variables:
              chl_default:
                short_name: chl
                mip: Oyr
                preprocessor: test
                timerange: '2000/2010'
                additional_datasets:
                  - project: CMIP5
                    dataset: CanESM2
                    exp: historical
                    ensemble: r1i1p1
            scripts: null
        """)
    get_recipe(tmp_path, content, session)


def test_invalid_stat_preproc(tmp_path, patched_datafinder, session):
    content = dedent("""
        preprocessors:
          test:
            invalid_statistics:
              operator: mean

        diagnostics:
          diagnostic_name:
            variables:
              chl_default:
                short_name: chl
                mip: Oyr
                preprocessor: test
                timerange: '2000/2010'
                additional_datasets:
                  - project: CMIP5
                    dataset: CanESM2
                    exp: historical
                    ensemble: r1i1p1
            scripts: null
        """)
    msg = "Unknown preprocessor function"
    with pytest.raises(ValueError, match=msg):
        get_recipe(tmp_path, content, session)


def test_bias_no_ref(tmp_path, patched_datafinder, session):
    content = dedent("""
        preprocessors:
          test_bias:
            bias:
              bias_type: relative
              denominator_mask_threshold: 5

        diagnostics:
          diagnostic_name:
            variables:
              ta:
                preprocessor: test_bias
                project: CMIP6
                mip: Amon
                exp: historical
                timerange: '20000101/20001231'
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
                  - {dataset: CESM2}

            scripts: null
        """)
    msg = (
        "Expected exactly 1 dataset with 'reference_for_bias: true' in "
        "products"
    )
    with pytest.raises(RecipeError) as exc:
        get_recipe(tmp_path, content, session)
    assert str(exc.value) == INITIALIZATION_ERROR_MSG
    assert msg in exc.value.failed_tasks[0].message
    assert "found 0" in exc.value.failed_tasks[0].message


def test_bias_two_refs(tmp_path, patched_datafinder, session):
    content = dedent("""
        preprocessors:
          test_bias:
            bias:
              bias_type: relative
              denominator_mask_threshold: 5

        diagnostics:
          diagnostic_name:
            variables:
              ta:
                preprocessor: test_bias
                project: CMIP6
                mip: Amon
                exp: historical
                timerange: '20000101/20001231'
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5, reference_for_bias: true}
                  - {dataset: CESM2, reference_for_bias: true}

            scripts: null
        """)
    msg = (
        "Expected exactly 1 dataset with 'reference_for_bias: true' in "
        "products"
    )
    with pytest.raises(RecipeError) as exc:
        get_recipe(tmp_path, content, session)
    assert str(exc.value) == INITIALIZATION_ERROR_MSG
    assert msg in exc.value.failed_tasks[0].message
    assert "found 2" in exc.value.failed_tasks[0].message


def test_invalid_bias_type(tmp_path, patched_datafinder, session):
    content = dedent("""
        preprocessors:
          test_bias:
            bias:
              bias_type: INVALID
              denominator_mask_threshold: 5

        diagnostics:
          diagnostic_name:
            variables:
              ta:
                preprocessor: test_bias
                project: CMIP6
                mip: Amon
                exp: historical
                timerange: '20000101/20001231'
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
                  - {dataset: CESM2, reference_for_bias: true}

            scripts: null
        """)
    msg = (
        "Expected one of ('absolute', 'relative') for `bias_type`, got "
        "'INVALID'"
    )
    with pytest.raises(RecipeError) as exc:
        get_recipe(tmp_path, content, session)
    assert str(exc.value) == INITIALIZATION_ERROR_MSG
    assert exc.value.failed_tasks[0].message == msg


def test_invalid_builtin_regridding_scheme(
        tmp_path, patched_datafinder, session
):
    content = dedent("""
        preprocessors:
          test:
            regrid:
              scheme: INVALID
              target_grid: 2x2
        diagnostics:
          diagnostic_name:
            variables:
              tas:
                mip: Amon
                preprocessor: test
                timerange: '2000/2010'
                additional_datasets:
                  - {project: CMIP5, dataset: CanESM2, exp: amip,
                     ensemble: r1i1p1}
            scripts: null
        """)
    msg = (
        "Got invalid built-in regridding scheme 'INVALID', expected one of "
    )
    with pytest.raises(RecipeError) as rec_err_exp:
        get_recipe(tmp_path, content, session)
    assert str(rec_err_exp.value) == INITIALIZATION_ERROR_MSG
    assert msg in str(rec_err_exp.value.failed_tasks[0].message)


def test_generic_regridding_scheme_no_ref(
        tmp_path, patched_datafinder, session
):
    content = dedent("""
        preprocessors:
          test:
            regrid:
              scheme:
                no_reference: given
              target_grid: 2x2
        diagnostics:
          diagnostic_name:
            variables:
              tas:
                mip: Amon
                preprocessor: test
                timerange: '2000/2010'
                additional_datasets:
                  - {project: CMIP5, dataset: CanESM2, exp: amip,
                     ensemble: r1i1p1}
            scripts: null
        """)
    msg = (
        "Failed to load generic regridding scheme: No reference specified for "
        "generic regridding. See "
    )
    with pytest.raises(RecipeError) as rec_err_exp:
        get_recipe(tmp_path, content, session)
    assert str(rec_err_exp.value) == INITIALIZATION_ERROR_MSG
    assert msg in str(rec_err_exp.value.failed_tasks[0].message)


def test_invalid_generic_regridding_scheme(
        tmp_path, patched_datafinder, session
):
    content = dedent("""
        preprocessors:
          test:
            regrid:
              scheme:
                reference: invalid.module:and.function
              target_grid: 2x2
        diagnostics:
          diagnostic_name:
            variables:
              tas:
                mip: Amon
                preprocessor: test
                timerange: '2000/2010'
                additional_datasets:
                  - {project: CMIP5, dataset: CanESM2, exp: amip,
                     ensemble: r1i1p1}
            scripts: null
        """)
    msg = (
        "Failed to load generic regridding scheme: Could not import specified "
        "generic regridding module 'invalid.module'. Please double check "
        "spelling and that the required module is installed. "
    )
    with pytest.raises(RecipeError) as rec_err_exp:
        get_recipe(tmp_path, content, session)
    assert str(rec_err_exp.value) == INITIALIZATION_ERROR_MSG
    assert msg in str(rec_err_exp.value.failed_tasks[0].message)


def test_deprecated_linear_extrapolate_scheme(
    tmp_path, patched_datafinder, session
):
    content = dedent("""
        preprocessors:
          test:
            regrid:
              scheme: linear_extrapolate
              target_grid: 2x2
        diagnostics:
          diagnostic_name:
            variables:
              tas:
                mip: Amon
                preprocessor: test
                timerange: '2000/2010'
                additional_datasets:
                  - {project: CMIP5, dataset: CanESM2, exp: amip,
                     ensemble: r1i1p1}
            scripts: null
        """)
    get_recipe(tmp_path, content, session)


def test_deprecated_unstructured_nearest_scheme(
    tmp_path, patched_datafinder, session
):
    content = dedent("""
        preprocessors:
          test:
            regrid:
              scheme: unstructured_nearest
              target_grid: 2x2
        diagnostics:
          diagnostic_name:
            variables:
              tas:
                mip: Amon
                preprocessor: test
                timerange: '2000/2010'
                additional_datasets:
                  - {project: CMIP5, dataset: CanESM2, exp: amip,
                     ensemble: r1i1p1}
            scripts: null
        """)
    get_recipe(tmp_path, content, session)


def test_wildcard_derived_var(
    tmp_path, patched_failing_datafinder, session
):
    content = dedent("""
        diagnostics:
          diagnostic_name:
            variables:
              swcre:
                mip: Amon
                derive: true
                force_derivation: true
                timerange: '2000/2010'
                additional_datasets:
                  - {project: CMIP5, dataset: '*', institute: '*', exp: amip,
                     ensemble: r1i1p1}
            scripts: null
        """)
    recipe = get_recipe(tmp_path, content, session)

    assert len(recipe.datasets) == 1
    dataset = recipe.datasets[0]
    assert dataset.facets['dataset'] == 'BBB'
    assert dataset.facets['institute'] == 'B'
    assert dataset.facets['short_name'] == 'swcre'


def test_distance_metric_no_ref(tmp_path, patched_datafinder, session):
    content = dedent("""
        preprocessors:
          test_distance_metric:
            distance_metric:
              metric: emd

        diagnostics:
          diagnostic_name:
            variables:
              ta:
                preprocessor: test_distance_metric
                project: CMIP6
                mip: Amon
                exp: historical
                timerange: '20000101/20001231'
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
                  - {dataset: CESM2}

            scripts: null
        """)
    msg = (
        "Expected exactly 1 dataset with 'reference_for_metric: true' in "
        "products"
    )
    with pytest.raises(RecipeError) as exc:
        get_recipe(tmp_path, content, session)
    assert str(exc.value) == INITIALIZATION_ERROR_MSG
    assert msg in exc.value.failed_tasks[0].message
    assert "found 0" in exc.value.failed_tasks[0].message


def test_distance_metric_two_refs(tmp_path, patched_datafinder, session):
    content = dedent("""
        preprocessors:
          test_distance_metric:
            distance_metric:
              metric: emd

        diagnostics:
          diagnostic_name:
            variables:
              ta:
                preprocessor: test_distance_metric
                project: CMIP6
                mip: Amon
                exp: historical
                timerange: '20000101/20001231'
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5, reference_for_metric: true}
                  - {dataset: CESM2, reference_for_metric: true}

            scripts: null
        """)
    msg = (
        "Expected exactly 1 dataset with 'reference_for_metric: true' in "
        "products"
    )
    with pytest.raises(RecipeError) as exc:
        get_recipe(tmp_path, content, session)
    assert str(exc.value) == INITIALIZATION_ERROR_MSG
    assert msg in exc.value.failed_tasks[0].message
    assert "found 2" in exc.value.failed_tasks[0].message


def test_invalid_metric(tmp_path, patched_datafinder, session):
    content = dedent("""
        preprocessors:
          test_distance_metric:
            distance_metric:
              metric: INVALID

        diagnostics:
          diagnostic_name:
            variables:
              ta:
                preprocessor: test_distance_metric
                project: CMIP6
                mip: Amon
                exp: historical
                timerange: '20000101/20001231'
                ensemble: r1i1p1f1
                grid: gn
                additional_datasets:
                  - {dataset: CanESM5}
                  - {dataset: CESM2, reference_for_metric: true}

            scripts: null
        """)
    msg = (
        "Expected one of ('rmse', 'weighted_rmse', 'pearsonr', "
        "'weighted_pearsonr', 'emd', 'weighted_emd') for `metric`, got "
        "'INVALID'"
    )
    with pytest.raises(RecipeError) as exc:
        get_recipe(tmp_path, content, session)
    assert str(exc.value) == INITIALIZATION_ERROR_MSG
    assert exc.value.failed_tasks[0].message == msg
