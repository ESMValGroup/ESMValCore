from unittest import mock

import iris
import numpy as np
import pyesgf.search.results
import pytest

import esmvalcore.experimental.recipe_output
from esmvalcore import _recipe
from esmvalcore.esgf._download import ESGFFile
from esmvalcore.exceptions import RecipeError
from tests import PreprocessorFile


class MockRecipe(_recipe.Recipe):
    """Mocked Recipe class with simple constructor."""

    def __init__(self, cfg, diagnostics):
        """Simple constructor used for testing."""
        self._cfg = cfg
        self.diagnostics = diagnostics


class TestRecipe:

    def test_expand_ensemble(self):

        datasets = [
            {
                'dataset': 'XYZ',
                'ensemble': 'r(1:2)i(2:3)p(3:4)',
            },
        ]

        expanded = _recipe.Recipe._expand_tag(datasets, 'ensemble')

        ensembles = [
            'r1i2p3',
            'r1i2p4',
            'r1i3p3',
            'r1i3p4',
            'r2i2p3',
            'r2i2p4',
            'r2i3p3',
            'r2i3p4',
        ]
        for i, ensemble in enumerate(ensembles):
            assert expanded[i] == {'dataset': 'XYZ', 'ensemble': ensemble}

    def test_expand_subexperiment(self):

        datasets = [
            {
                'dataset': 'XYZ',
                'sub_experiment': 's(1998:2005)',
            },
        ]

        expanded = _recipe.Recipe._expand_tag(datasets, 'sub_experiment')

        subexperiments = [
            's1998',
            's1999',
            's2000',
            's2001',
            's2002',
            's2003',
            's2004',
            's2005',
        ]
        for i, subexperiment in enumerate(subexperiments):
            assert expanded[i] == {
                'dataset': 'XYZ',
                'sub_experiment': subexperiment
            }

    def test_expand_ensemble_nolist(self):

        datasets = [
            {
                'dataset': 'XYZ',
                'ensemble': ['r1i1p1', 'r(1:2)i1p1']
            },
        ]

        with pytest.raises(RecipeError):
            _recipe.Recipe._expand_tag(datasets, 'ensemble')


VAR_A = {'dataset': 'A'}
VAR_A_REF_A = {'dataset': 'A', 'reference_dataset': 'A'}
VAR_A_REF_B = {'dataset': 'A', 'reference_dataset': 'B'}

TEST_ALLOW_SKIPPING = [
    ([], VAR_A, {}, False),
    ([], VAR_A, {
        'skip-nonexistent': False
    }, False),
    ([], VAR_A, {
        'skip-nonexistent': True
    }, True),
    ([], VAR_A_REF_A, {}, False),
    ([], VAR_A_REF_A, {
        'skip-nonexistent': False
    }, False),
    ([], VAR_A_REF_A, {
        'skip-nonexistent': True
    }, False),
    ([], VAR_A_REF_B, {}, False),
    ([], VAR_A_REF_B, {
        'skip-nonexistent': False
    }, False),
    ([], VAR_A_REF_B, {
        'skip-nonexistent': True
    }, True),
    (['A'], VAR_A, {}, False),
    (['A'], VAR_A, {
        'skip-nonexistent': False
    }, False),
    (['A'], VAR_A, {
        'skip-nonexistent': True
    }, False),
    (['A'], VAR_A_REF_A, {}, False),
    (['A'], VAR_A_REF_A, {
        'skip-nonexistent': False
    }, False),
    (['A'], VAR_A_REF_A, {
        'skip-nonexistent': True
    }, False),
    (['A'], VAR_A_REF_B, {}, False),
    (['A'], VAR_A_REF_B, {
        'skip-nonexistent': False
    }, False),
    (['A'], VAR_A_REF_B, {
        'skip-nonexistent': True
    }, False),
]


@pytest.mark.parametrize('ancestors,var,cfg,out', TEST_ALLOW_SKIPPING)
def test_allow_skipping(ancestors, var, cfg, out):
    """Test ``_allow_skipping``."""
    result = _recipe._allow_skipping(ancestors, var, cfg)
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
    recipe._cfg = {
        'resume_from': [str(prev_output)],
        'preproc_dir': '/path/to/recipe_test_20210101_000000/preproc',
    }

    # Create a very simplified list of datasets
    diagnostic = {'preprocessor_output': {'tas': [{'short_name': 'tas'}]}}

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
@pytest.mark.parametrize('already_downloaded', [True, False])
def test_search_esgf(mocker, tmp_path, local_availability, already_downloaded):

    rootpath = tmp_path / 'local'
    download_dir = tmp_path / 'download_dir'
    esgf_files = create_esgf_search_results()

    # ESGF files may have been downloaded previously, but not have
    # been found if the download_dir is not configured as a rootpath
    if already_downloaded:
        for file in esgf_files:
            local_path = file.local_file(download_dir)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.touch()

    # Local files can cover the entire period, part of it, or nothing
    local_file_options = {
        'all': [f.local_file(rootpath).as_posix() for f in esgf_files],
        'partial': [esgf_files[1].local_file(rootpath).as_posix()],
        'none': [],
    }
    local_files = local_file_options[local_availability]

    mocker.patch.object(_recipe,
                        'get_input_filelist',
                        autospec=True,
                        return_value=(list(local_files), [], []))
    mocker.patch.object(
        _recipe.esgf,
        'find_files',
        autospec=True,
        return_value=esgf_files,
    )

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

    config_user = {
        'rootpath': None,
        'drs': None,
        'offline': False,
        'download_dir': download_dir
    }
    input_files = _recipe._get_input_files(variable, config_user)[0]

    download_files = [
        f.local_file(download_dir).as_posix() for f in esgf_files
    ]

    expected = {
        'all': local_files,
        'partial': local_files + download_files[:1],
        'none': download_files,
    }
    assert input_files == expected[local_availability]


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


def test_add_fxvar_keys_extra_facets():
    """Test correct addition of extra facets to fx variables."""
    fx_info = {'short_name': 'areacella', 'mip': 'fx'}
    variable = {'project': 'ICON', 'dataset': 'ICON'}
    extra_facets_dir = tuple()
    fx_var = _recipe._add_fxvar_keys(fx_info, variable, extra_facets_dir)
    expected_fx_var = {
        # Already given by fx_info and variable
        'short_name': 'areacella',
        'mip': 'fx',
        'project': 'ICON',
        'dataset': 'ICON',
        # Added by _add_fxvar_keys
        'variable_group': 'areacella',
        # Added by _add_cmor_info
        'original_short_name': 'areacella',
        'standard_name': 'cell_area',
        'long_name': 'Grid-Cell Area for Atmospheric Grid Variables',
        'units': 'm2',
        'modeling_realm': ['atmos', 'land'],
        'frequency': 'fx',
        # Added by _add_extra_facets
        'latitude': 'grid_latitude',
        'longitude': 'grid_longitude',
        'raw_name': 'cell_area',
    }
    assert fx_var == expected_fx_var


def test_multi_model_filename():
    """Test timerange in multi-model filename is correct."""
    cube = iris.cube.Cube(np.array([1]))
    products = [
        PreprocessorFile(cube, 'A', {'timerange': '1990/1991'}),
        PreprocessorFile(cube, 'B', {'timerange': '1989/1990'}),
        PreprocessorFile(cube, 'C', {'timerange': '1991/1992'}),
    ]
    attributes = _recipe._get_statistic_attributes(products)
    assert 'timerange' in attributes
    assert attributes['timerange'] == '1989/1992'


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
    (None, []),
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
    cfg = {}
    if diags_to_run is not None:
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
