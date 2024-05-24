from pathlib import Path
from unittest import mock

import iris
import numpy as np
import pyesgf.search.results
import pytest

import esmvalcore
import esmvalcore._recipe.recipe as _recipe
import esmvalcore.config
import esmvalcore.experimental.recipe_output
from esmvalcore.dataset import Dataset
from esmvalcore.esgf._download import ESGFFile
from esmvalcore.exceptions import RecipeError
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
    dataset_id = (
        'CMIP6.CMIP.EC-Earth-Consortium.EC-Earth3.historical.r1i1p1f1'
        '.Amon.tas.gr.v20200310|esgf-data1.llnl.gov'
    )
    dataset_id_template = (
        '%(mip_era)s.%(activity_drs)s.%(institution_id)s.'
        '%(source_id)s.%(experiment_id)s.%(member_id)s.%(table_id)s.'
        '%(variable_id)s.%(grid_label)s'
    )
    file0 = ESGFFile([
        pyesgf.search.results.FileResult(
            json={
                'dataset_id': dataset_id,
                'dataset_id_template_': [dataset_id_template],
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
                'dataset_id': dataset_id,
                'dataset_id_template_': [dataset_id_template],
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
def test_schedule_for_download(monkeypatch, tmp_path, local_availability):
    """Test that `_schedule_for_download` updates DOWNLOAD_FILES."""
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
    _recipe._schedule_for_download([dataset])
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


def test_update_multiproduct_multi_model_statistics_percentile():
    """Test ``_update_multiproduct``."""
    settings = {
        'multi_model_statistics': {
            'statistics': [
                {'operator': 'percentile', 'percent': 5.0},
                {'operator': 'percentile', 'percent': 95.0},
            ]
        },
    }
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
    assert (
        Path('/preproc/d/var/CMIP6_MultiModelPercentile5-0_2002-2004.nc') in
        filenames
    )
    assert (
        Path('/preproc/d/var/CMIP6_MultiModelPercentile95-0_2002-2004.nc') in
        filenames
    )

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
        if 'MultiModelPercentile5-0' in str(product.filename):
            assert product.attributes['alias'] == 'MultiModelPercentile5-0'
            assert product.attributes['dataset'] == 'MultiModelPercentile5-0'
            assert (product.attributes['multi_model_statistics'] ==
                    'MultiModelPercentile5-0')
        elif 'MultiModelPercentile95-0' in str(product.filename):
            assert product.attributes['alias'] == 'MultiModelPercentile95-0'
            assert product.attributes['dataset'] == 'MultiModelPercentile95-0'
            assert (product.attributes['multi_model_statistics'] ==
                    'MultiModelPercentile95-0')

    assert len(settings) == 1
    output_products = settings['output_products']
    assert len(output_products) == 1
    stats = output_products['']
    assert len(stats) == 2
    assert 'percentile5.0' in stats
    assert 'percentile95.0' in stats
    assert 'MultiModelPercentile5-0' in str(stats['percentile5.0'].filename)
    assert 'MultiModelPercentile95-0' in str(stats['percentile95.0'].filename)


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
    assert product.filename == Path(
        '/preproc/d/var/CMIP6_CanESM2_EnsembleMedian_2000-2000.nc')

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
    assert stats['median'].filename == Path(
        '/preproc/d/var/CMIP6_CanESM2_EnsembleMedian_2000-2000.nc')


def test_update_multiproduct_ensemble_statistics_percentile():
    """Test ``_update_multiproduct``."""
    settings = {
        'ensemble_statistics': {
            'statistics': [
                {'operator': 'percentile', 'percent': 5},
            ],
            'span': 'full',
        },
    }

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
    assert product.filename == Path(
        '/preproc/d/var/CMIP6_CanESM2_EnsemblePercentile5_2000-2000.nc')

    for attr in common_attributes:
        assert attr in product.attributes
        assert product.attributes[attr] == common_attributes[attr]
        assert 'alias' in product.attributes
        assert product.attributes['alias'] == 'EnsemblePercentile5'
        assert 'dataset' in product.attributes
        assert product.attributes['dataset'] == 'CanESM2'
        assert 'ensemble_statistics' in product.attributes
        assert product.attributes['ensemble_statistics'] == (
            'EnsemblePercentile5'
        )
        assert 'start_year' in product.attributes
        assert product.attributes['start_year'] == 2000
        assert 'end_year' in product.attributes
        assert product.attributes['end_year'] == 2000

    assert len(settings) == 1
    output_products = settings['output_products']
    assert len(output_products) == 1
    stats = output_products['CMIP6_CanESM2']
    assert len(stats) == 1
    assert 'percentile5' in stats
    assert stats['percentile5'].filename == Path(
        '/preproc/d/var/CMIP6_CanESM2_EnsemblePercentile5_2000-2000.nc')


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
@mock.patch('esmvalcore._recipe.recipe.DiagnosticTask', autospec=True)
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


def test_update_regrid_time():
    """Test `_update_regrid_time."""
    dataset = Dataset(frequency='mon')
    settings = {'regrid_time': {}}
    _recipe._update_regrid_time(dataset, settings)
    assert settings == {'regrid_time': {'frequency': 'mon'}}


def test_select_dataset_fails():
    dataset = Dataset(
        dataset='dataset1',
        diagnostic='diagnostic1',
        variable_group='tas',
    )
    with pytest.raises(RecipeError):
        _recipe._select_dataset('dataset2', [dataset])


def test_limit_datasets():

    datasets = [
        Dataset(dataset='dataset1', alias='dataset1'),
        Dataset(dataset='dataset2', alias='dataset2'),
    ]
    datasets[0].session = {'max_datasets': 1}

    result = _recipe._limit_datasets(datasets, {})

    assert result == datasets[:1]


def test_get_default_settings(mocker):
    mocker.patch.object(
        _recipe,
        '_get_output_file',
        autospec=True,
        return_value=Path('/path/to/file.nc'),
    )
    session = mocker.create_autospec(esmvalcore.config.Session, instance=True)
    session.__getitem__.return_value = False

    dataset = Dataset(
        short_name='sic',
        original_short_name='siconc',
        mip='Amon',
        project='CMIP6',
    )
    dataset.session = session

    settings = _recipe._get_default_settings(dataset)
    assert settings == {
        'remove_supplementary_variables': {},
        'save': {'compress': False, 'alias': 'sic'},
    }


def test_set_version(mocker):

    dataset = Dataset(short_name='tas')
    supplementary = Dataset(short_name='areacella')
    dataset.supplementaries = [supplementary]

    input_dataset = Dataset(short_name='tas')
    file1 = mocker.Mock()
    file1.facets = {'version': 'v1'}
    file2 = mocker.Mock()
    file2.facets = {'version': 'v2'}
    input_dataset.files = [file1, file2]

    file3 = mocker.Mock()
    file3.facets = {'version': 'v3'}
    supplementary.files = [file3]

    _recipe._set_version(dataset, [input_dataset])
    print(dataset)
    assert dataset.facets['version'] == ['v1', 'v2']
    assert dataset.supplementaries[0].facets['version'] == 'v3'


def test_extract_preprocessor_order():
    profile = {
        'custom_order': True,
        'regrid': {
            'target_grid': '1x1'
        },
        'derive': {
            'long_name': 'albedo at the surface',
            'short_name': 'alb',
            'standard_name': '',
            'units': '1'
        },
    }
    order = _recipe._extract_preprocessor_order(profile)
    assert any(order[i:i + 2] == ('regrid', 'derive')
               for i in range(len(order) - 1))


def test_update_extract_shape_abs_shapefile(session, tmp_path):
    """Test ``_update_extract_shape``."""
    session['auxiliary_data_dir'] = '/aux/dir'
    shapefile = tmp_path / 'my_custom_shapefile.shp'
    shapefile.write_text("")  # create empty file
    settings = {'extract_shape': {'shapefile': str(shapefile)}}

    _recipe._update_extract_shape(settings, session)

    assert isinstance(settings['extract_shape']['shapefile'], Path)
    assert settings['extract_shape']['shapefile'] == shapefile


@pytest.mark.parametrize(
    'shapefile', ['aux_dir/ar6.shp', 'ar6.shp', 'ar6', 'AR6', 'aR6']
)
def test_update_extract_shape_rel_shapefile(shapefile, session, tmp_path):
    """Test ``_update_extract_shape``."""
    session['auxiliary_data_dir'] = tmp_path
    (tmp_path / 'aux_dir').mkdir(parents=True, exist_ok=True)
    aux_dir_shapefile = tmp_path / 'aux_dir' / 'ar6.shp'
    aux_dir_shapefile.write_text("")  # create empty file
    settings = {'extract_shape': {'shapefile': shapefile}}

    _recipe._update_extract_shape(settings, session)

    if 'aux_dir' in shapefile:
        assert settings['extract_shape']['shapefile'] == tmp_path / shapefile
    else:
        ar6_file = (
            Path(esmvalcore.preprocessor.__file__).parent / 'shapefiles' /
            'ar6.shp'
        )
        assert settings['extract_shape']['shapefile'] == ar6_file
