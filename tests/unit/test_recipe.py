from pathlib import Path
from unittest import mock

import iris
import numpy as np
import pyesgf.search.results
import pytest

import esmvalcore.experimental.recipe_output
from esmvalcore import _recipe
from esmvalcore.dataset import Dataset
from esmvalcore.esgf._download import ESGFFile
from tests import PreprocessorFile


class MockRecipe(_recipe.Recipe):
    """Mocked Recipe class with simple constructor."""

    def __init__(self, cfg, diagnostics):
        """Simple constructor used for testing."""
        self.session = cfg
        self.diagnostics = diagnostics


VAR_A = {'dataset': 'A'}
VAR_A_REF_A = {'dataset': 'A', 'reference_dataset': 'A'}
VAR_A_REF_B = {'dataset': 'A', 'reference_dataset': 'B'}

TEST_ALLOW_SKIPPING = [
    (VAR_A, {
        'skip_nonexistent': False
    }, False),
    (VAR_A, {
        'skip_nonexistent': True
    }, True),
    (VAR_A_REF_A, {
        'skip_nonexistent': False
    }, False),
    (VAR_A_REF_A, {
        'skip_nonexistent': True
    }, False),
    (VAR_A_REF_B, {
        'skip_nonexistent': False
    }, False),
    (VAR_A_REF_B, {
        'skip_nonexistent': True
    }, True),
]


@pytest.mark.parametrize('var,cfg,out', TEST_ALLOW_SKIPPING)
def test_allow_skipping(var, cfg, out):
    """Test ``_allow_skipping``."""
    dataset = Dataset(**var)
    dataset.session = cfg
    result = _recipe._allow_skipping(dataset)
    assert result is out


def test_resume_preprocessor_tasks(mocker, tmp_path):
    """Test that `Recipe._create_preprocessor_tasks` creates a ResumeTask."""
    # Create a mock ResumeTask class that returns a mock instance
    resume_task_cls = mocker.patch.object(_recipe, 'ResumeTask', autospec=True)
    resume_task = mocker.Mock()
    resume_task_cls.return_value = resume_task

    # Create a mock output directory of a previous run
    diagnostic_name = 'diagnostic_name'
    prev_output = tmp_path / 'recipe_test_20200101_000000'
    prev_preproc_dir = prev_output / 'preproc' / diagnostic_name / 'tas'
    prev_preproc_dir.mkdir(parents=True)

    # Create a mock recipe
    recipe = mocker.create_autospec(_recipe.Recipe, instance=True)

    class Session(dict):
        pass
    session = Session(resume_from=[prev_output])
    session.preproc_dir = Path('/path/to/recipe_test_20210101_000000/preproc')
    recipe.session = session

    # Create a very simplified list of datasets
    diagnostic = {
        'datasets': [Dataset(short_name='tas', variable_group='tas')],
    }

    # Create tasks
    tasks, failed = _recipe.Recipe._create_preprocessor_tasks(
        recipe, diagnostic_name, diagnostic, [], True)

    assert tasks == [resume_task]
    assert not failed


def create_esgf_search_results():
    """Prepare some fake ESGF search results."""
    file0 = ESGFFile([
        pyesgf.search.results.FileResult(
            json={
                'dataset_id':
                'CMIP6.CMIP.EC-Earth-Consortium.EC-Earth3.historical.r1i1p1f1'
                '.Amon.tas.gr.v20200310|esgf-data1.llnl.gov',
                'project': ['CMIP6'],
                'size':
                4745571,
                'source_id': ['EC-Earth3'],
                'title':
                'tas_Amon_EC-Earth3_historical_r1i1p1f1_gr_185001-185012.nc',
                'url': [
                    'http://esgf-data1.llnl.gov/thredds/fileServer/css03_data'
                    '/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/historical'
                    '/r1i1p1f1/Amon/tas/gr/v20200310/tas_Amon_EC-Earth3'
                    '_historical_r1i1p1f1_gr_185001-185012.nc'
                    '|application/netcdf|HTTPServer',
                ],
            },
            context=None,
        )
    ])
    file1 = ESGFFile([
        pyesgf.search.results.FileResult(
            {
                'dataset_id':
                'CMIP6.CMIP.EC-Earth-Consortium.EC-Earth3.historical.r1i1p1f1'
                '.Amon.tas.gr.v20200310|esgf-data1.llnl.gov',
                'project': ['CMIP6'],
                'size':
                4740192,
                'source_id': ['EC-Earth3'],
                'title':
                'tas_Amon_EC-Earth3_historical_r1i1p1f1_gr_185101-185112.nc',
                'url': [
                    'http://esgf-data1.llnl.gov/thredds/fileServer/css03_data'
                    '/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/historical'
                    '/r1i1p1f1/Amon/tas/gr/v20200310/tas_Amon_EC-Earth3'
                    '_historical_r1i1p1f1_gr_185101-185112.nc'
                    '|application/netcdf|HTTPServer',
                ],
            },
            context=None,
        )
    ])

    return [file0, file1]


@pytest.mark.parametrize("local_availability", ['all', 'partial', 'none'])
def test_check_input_files(monkeypatch, tmp_path, local_availability):
    """Test _check_input_files: it does not raise and updates
    DOWNLOAD_FILES."""
    esgf_files = create_esgf_search_results()
    download_dir = tmp_path / 'download_dir'
    local_dir = Path('/local_dir')

    # Local files can cover the entire period, part of it, or nothing
    local_file_options = {
        'all': [f.local_file(local_dir) for f in esgf_files],
        'partial': [esgf_files[1].local_file(local_dir)],
        'none': [],
    }
    local_files = local_file_options[local_availability]

    variable = {
        'project': 'CMIP6',
        'mip': 'Amon',
        'frequency': 'mon',
        'short_name': 'tas',
        'dataset': 'EC.-Earth3',
        'exp': 'historical',
        'ensemble': 'r1i1p1f1',
        'grid': 'gr',
        'timerange': '1850/1851',
        'alias': 'CMIP6_EC-Eeath3_tas',
    }
    dataset = Dataset(**variable)
    files = {
        'all': local_files,
        'partial': local_files + esgf_files[:1],
        'none': esgf_files,
    }
    dataset.session = {'download_dir': download_dir}
    dataset.files = list(files[local_availability])

    monkeypatch.setattr(_recipe, 'DOWNLOAD_FILES', set())
    _recipe._check_input_files(dataset)
    print(esgf_files)
    expected = {
        'all': set(),
        'partial': set(esgf_files[:1]),
        'none': set(esgf_files),
    }
    assert _recipe.DOWNLOAD_FILES == expected[local_availability]


def test_write_html_summary(mocker, caplog):
    """Test `Recipe.write_html_summary` failing and logging a message."""
    message = "Failed to look up references."
    recipe_output = mocker.patch.object(
        esmvalcore.experimental.recipe_output,
        'RecipeOutput',
        create_autospec=True,
    )
    recipe_output.from_core_recipe_output.side_effect = LookupError(message)
    mock_recipe = mocker.create_autospec(_recipe.Recipe, instance=True)
    caplog.set_level('WARNING')

    _recipe.Recipe.write_html_summary(mock_recipe)

    assert f"Could not write HTML report: {message}" in caplog.text
    mock_recipe.get_output.assert_called_once()


def test_multi_model_filename_overlap():
    """Test timerange in multi-model filename is correct."""
    cube = iris.cube.Cube(np.array([1]))
    products = [
        PreprocessorFile(cube, 'A', {'timerange': '19900101/19911010'}),
        PreprocessorFile(cube, 'B', {'timerange': '19891212/19910505'}),
        PreprocessorFile(cube, 'C', {'timerange': '19910202/19921111'}),
    ]
    settings = {}  # the default setting for "span" is "overlap"
    attributes = _recipe._get_common_attributes(products, settings)
    assert 'timerange' in attributes
    assert attributes['timerange'] == '19910202/19910505'
    assert attributes['start_year'] == 1991
    assert attributes['end_year'] == 1991


def test_multi_model_filename_full():
    """Test timerange in multi-model filename is correct."""
    cube = iris.cube.Cube(np.array([1]))
    products = [
        PreprocessorFile(cube, 'A', {'timerange': '19900101/19911010'}),
        PreprocessorFile(cube, 'B', {'timerange': '19891212/19910505'}),
        PreprocessorFile(cube, 'C', {'timerange': '19910202/19921111'}),
    ]
    settings = {'span': 'full'}
    attributes = _recipe._get_common_attributes(products, settings)
    assert 'timerange' in attributes
    assert attributes['timerange'] == '19891212/19921111'
    assert attributes['start_year'] == 1989
    assert attributes['end_year'] == 1992


def test_update_multiproduct_multi_model_statistics():
    """Test ``_update_multiproduct``."""
    settings = {'multi_model_statistics': {'statistics': ['mean', 'std_dev']}}
    common_attributes = {
        'project': 'CMIP6',
        'diagnostic': 'd',
        'variable_group': 'var',
    }
    cube = iris.cube.Cube(np.array([1]))
    products = [
        PreprocessorFile(cube, 'A',
                         attributes={'dataset': 'a',
                                     'timerange': '2000/2005',
                                     **common_attributes},
                         settings=settings),
        PreprocessorFile(cube, 'B',
                         attributes={'dataset': 'b',
                                     'timerange': '2001/2004',
                                     **common_attributes},
                         settings=settings),
        PreprocessorFile(cube, 'C',
                         attributes={'dataset': 'c',
                                     'timerange': '1999/2004',
                                     **common_attributes},
                         settings=settings),
        PreprocessorFile(cube, 'D',
                         attributes={'dataset': 'd',
                                     'timerange': '2002/2010',
                                     **common_attributes},
                         settings=settings),
    ]
    order = ('load', 'multi_model_statistics', 'save')
    preproc_dir = '/preproc'
    step = 'multi_model_statistics'
    output, settings = _recipe._update_multiproduct(
        products, order, preproc_dir, step)

    assert len(output) == 2

    filenames = [p.filename for p in output]
    assert Path(
        '/preproc/d/var/CMIP6_MultiModelMean_2002-2004.nc') in filenames
    assert Path(
        '/preproc/d/var/CMIP6_MultiModelStd_Dev_2002-2004.nc') in filenames

    for product in output:
        for attr in common_attributes:
            assert attr in product.attributes
            assert product.attributes[attr] == common_attributes[attr]
            assert 'alias' in product.attributes
            assert 'dataset' in product.attributes
            assert 'multi_model_statistics' in product.attributes
            assert 'timerange' in product.attributes
            assert product.attributes['timerange'] == '2002/2004'
            assert 'start_year' in product.attributes
            assert product.attributes['start_year'] == 2002
            assert 'end_year' in product.attributes
            assert product.attributes['end_year'] == 2004
        if 'MultiModelStd_Dev' in str(product.filename):
            assert product.attributes['alias'] == 'MultiModelStd_Dev'
            assert product.attributes['dataset'] == 'MultiModelStd_Dev'
            assert (product.attributes['multi_model_statistics'] ==
                    'MultiModelStd_Dev')
        elif 'MultiModelMean' in str(product.filename):
            assert product.attributes['alias'] == 'MultiModelMean'
            assert product.attributes['dataset'] == 'MultiModelMean'
            assert (product.attributes['multi_model_statistics'] ==
                    'MultiModelMean')

    assert len(settings) == 1
    output_products = settings['output_products']
    assert len(output_products) == 1
    stats = output_products['']
    assert len(stats) == 2
    assert 'mean' in stats
    assert 'std_dev' in stats
    assert 'MultiModelMean' in str(stats['mean'].filename)
    assert 'MultiModelStd_Dev' in str(stats['std_dev'].filename)


def test_update_multiproduct_ensemble_statistics():
    """Test ``_update_multiproduct``."""
    settings = {'ensemble_statistics': {'statistics': ['median'],
                                        'span': 'full'}}
    common_attributes = {
        'dataset': 'CanESM2',
        'project': 'CMIP6',
        'timerange': '2000/2000',
        'diagnostic': 'd',
        'variable_group': 'var',
    }
    cube = iris.cube.Cube(np.array([1]))
    products = [
        PreprocessorFile(cube, 'A',
                         attributes=common_attributes,
                         settings=settings),
        PreprocessorFile(cube, 'B',
                         attributes=common_attributes,
                         settings=settings),
        PreprocessorFile(cube, 'C',
                         attributes=common_attributes,
                         settings=settings),
        PreprocessorFile(cube, 'D',
                         attributes=common_attributes,
                         settings=settings),
    ]
    order = ('load', 'ensemble_statistics', 'save')
    preproc_dir = '/preproc'
    step = 'ensemble_statistics'
    output, settings = _recipe._update_multiproduct(
        products, order, preproc_dir, step)

    assert len(output) == 1
    product = list(output)[0]
    assert (product.filename == Path(
        '/preproc/d/var/CMIP6_CanESM2_EnsembleMedian_2000-2000.nc'))

    for attr in common_attributes:
        assert attr in product.attributes
        assert product.attributes[attr] == common_attributes[attr]
        assert 'alias' in product.attributes
        assert product.attributes['alias'] == 'EnsembleMedian'
        assert 'dataset' in product.attributes
        assert product.attributes['dataset'] == 'CanESM2'
        assert 'ensemble_statistics' in product.attributes
        assert product.attributes['ensemble_statistics'] == 'EnsembleMedian'
        assert 'start_year' in product.attributes
        assert product.attributes['start_year'] == 2000
        assert 'end_year' in product.attributes
        assert product.attributes['end_year'] == 2000

    assert len(settings) == 1
    output_products = settings['output_products']
    assert len(output_products) == 1
    stats = output_products['CMIP6_CanESM2']
    assert len(stats) == 1
    assert 'median' in stats
    assert (stats['median'].filename == Path(
        '/preproc/d/var/CMIP6_CanESM2_EnsembleMedian_2000-2000.nc'))


def test_update_multiproduct_no_product():
    cube = iris.cube.Cube(np.array([1]))
    products = [
        PreprocessorFile(cube, 'A', attributes=None, settings={'step': {}})]
    order = ('load', 'save')
    preproc_dir = '/preproc_dir'
    step = 'multi_model_statistics'
    output, settings = _recipe._update_multiproduct(
        products, order, preproc_dir, step)
    assert output == products
    assert settings == {}


SCRIPTS_CFG = {
    'output_dir': mock.sentinel.output_dir,
    'script': mock.sentinel.script,
    'settings': mock.sentinel.settings,
}
DIAGNOSTICS = {
    'd1': {'scripts': {'s1': {'ancestors': [], **SCRIPTS_CFG}}},
    'd2': {'scripts': {'s1': {'ancestors': ['d1/pr', 'd1/s1'],
                              **SCRIPTS_CFG}}},
    'd3': {'scripts': {'s1': {'ancestors': ['d2/s1'], **SCRIPTS_CFG}}},
    'd4': {'scripts': {
        's1': {'ancestors': 'd1/pr d1/tas', **SCRIPTS_CFG},
        's2': {'ancestors': ['d4/pr', 'd4/tas'], **SCRIPTS_CFG},
        's3': {'ancestors': ['d3/s1'], **SCRIPTS_CFG},
    }},
}
TEST_GET_TASKS_TO_RUN = [
    (None, None),
    ({''}, {''}),
    ({'wrong_task/*'}, {'wrong_task/*'}),
    ({'d1/*'}, {'d1/*'}),
    ({'d2/*'}, {'d2/*', 'd1/pr', 'd1/s1'}),
    ({'d3/*'}, {'d3/*', 'd2/s1', 'd1/pr', 'd1/s1'}),
    ({'d4/*'}, {'d4/*', 'd1/pr', 'd1/tas', 'd4/pr', 'd4/tas', 'd3/s1',
                'd2/s1', 'd1/s1'}),
    ({'wrong_task/*', 'd1/*'}, {'wrong_task/*', 'd1/*'}),
    ({'d1/ta'}, {'d1/ta'}),
    ({'d4/s2'}, {'d4/s2', 'd4/pr', 'd4/tas'}),
    ({'d2/s1', 'd3/ta', 'd1/s1'}, {'d2/s1', 'd1/pr', 'd1/s1', 'd3/ta'}),
    ({'d4/s1', 'd4/s2'}, {'d4/s1', 'd1/pr', 'd1/tas', 'd4/s2', 'd4/pr',
                          'd4/tas'}),
    ({'d4/s3', 'd3/ta'}, {'d4/s3', 'd3/s1', 'd2/s1', 'd1/pr', 'd1/s1',
                          'd3/ta'}),
]


@pytest.mark.parametrize('diags_to_run,tasknames_to_run',
                         TEST_GET_TASKS_TO_RUN)
def test_get_tasks_to_run(diags_to_run, tasknames_to_run):
    """Test ``Recipe._get_tasks_to_run``."""
    cfg = {'diagnostics': diags_to_run}

    recipe = MockRecipe(cfg, DIAGNOSTICS)
    tasks_to_run = recipe._get_tasks_to_run()

    assert tasks_to_run == tasknames_to_run


TEST_CREATE_DIAGNOSTIC_TASKS = [
    (set(), ['s1', 's2', 's3']),
    ({'d4/*'}, ['s1', 's2', 's3']),
    ({'d4/s1'}, ['s1']),
    ({'d4/s1', 'd3/*'}, ['s1']),
    ({'d4/s1', 'd4/s2'}, ['s1', 's2']),
    ({''}, []),
    ({'d3/*'}, []),
]


@pytest.mark.parametrize('tasks_to_run,tasks_run',
                         TEST_CREATE_DIAGNOSTIC_TASKS)
@mock.patch('esmvalcore._recipe.DiagnosticTask', autospec=True)
def test_create_diagnostic_tasks(mock_diag_task, tasks_to_run, tasks_run):
    """Test ``Recipe._create_diagnostic_tasks``."""
    cfg = {'run_diagnostic': True}
    diag_name = 'd4'
    diag_cfg = DIAGNOSTICS['d4']
    n_tasks = len(tasks_run)

    recipe = MockRecipe(cfg, DIAGNOSTICS)
    tasks = recipe._create_diagnostic_tasks(diag_name, diag_cfg, tasks_to_run)

    assert len(tasks) == n_tasks
    assert mock_diag_task.call_count == n_tasks
    for task_name in tasks_run:
        expected_call = mock.call(
            script=mock.sentinel.script,
            output_dir=mock.sentinel.output_dir,
            settings=mock.sentinel.settings,
            name=f'{diag_name}{_recipe.TASKSEP}{task_name}',
        )
        assert expected_call in mock_diag_task.mock_calls


def test_differing_timeranges(caplog):
    timeranges = set()
    timeranges.add('1950/1951')
    timeranges.add('1950/1952')
    required_variables = [
        {
            'short_name': 'rsdscs',
            'timerange': '1950/1951'
        },
        {
            'short_name': 'rsuscs',
            'timerange': '1950/1952'
        },
    ]
    with pytest.raises(ValueError) as exc:
        _recipe._check_differing_timeranges(
            timeranges, required_variables)
    expected_log = (
        f"Differing timeranges with values {timeranges} "
        "found for required variables "
        "[{'short_name': 'rsdscs', 'timerange': '1950/1951'}, "
        "{'short_name': 'rsuscs', 'timerange': '1950/1952'}]. "
        "Set `timerange` to a common value."
    )

    assert expected_log in str(exc.value)


def test_update_warning_settings_nonaffected_project():
    """Test ``_update_warning_settings``."""
    settings = {'save': {'filename': 'out.nc'}, 'load': {'filename': 'in.nc'}}
    _recipe._update_warning_settings(settings, 'CMIP5')
    assert settings == {
        'save': {'filename': 'out.nc'},
        'load': {'filename': 'in.nc'},
    }


def test_update_warning_settings_step_not_present():
    """Test ``_update_warning_settings``."""
    settings = {'save': {'filename': 'out.nc'}}
    _recipe._update_warning_settings(settings, 'EMAC')
    assert settings == {'save': {'filename': 'out.nc'}}


def test_update_warning_settings_step_present():
    """Test ``_update_warning_settings``."""
    settings = {'save': {'filename': 'out.nc'}, 'load': {'filename': 'in.nc'}}
    _recipe._update_warning_settings(settings, 'EMAC')
    assert len(settings) == 2
    assert settings['save'] == {'filename': 'out.nc'}
    assert len(settings['load']) == 2
    assert settings['load']['filename'] == 'in.nc'
    assert 'ignore_warnings' in settings['load']
