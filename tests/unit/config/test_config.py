import textwrap
from pathlib import Path

import pytest
import yaml

from esmvalcore.cmor.check import CheckLevels
from esmvalcore.config import CFG, _config, _config_validators
from esmvalcore.config._config import (
    _deep_update,
    _load_extra_facets,
    get_extra_facets,
    get_ignored_warnings,
    importlib_files,
)
from esmvalcore.dataset import Dataset
from esmvalcore.exceptions import ESMValCoreDeprecationWarning, RecipeError

TEST_DEEP_UPDATE = [
    ([{}], {}),
    ([{"a": 1, "b": 2}, {"a": 3}], {"a": 3, "b": 2}),
    (
        [
            {"a": {"b": 1, "c": {"d": 2}}, "e": {"f": 4, "g": 5}},
            {"a": {"b": 2, "c": 3}},
        ],
        {"a": {"b": 2, "c": 3}, "e": {"f": 4, "g": 5}},
    ),
]


@pytest.mark.parametrize(("dictionaries", "expected_merged"), TEST_DEEP_UPDATE)
def test_deep_update(dictionaries, expected_merged):
    merged = dictionaries[0]
    for update in dictionaries[1:]:
        merged = _deep_update(merged, update)
    assert expected_merged == merged


BASE_PATH = importlib_files("tests")
BASE_PATH /= Path("sample_data") / Path("extra_facets")  # type: ignore

TEST_LOAD_EXTRA_FACETS = [
    ("test-nonexistent", (), {}),
    ("test-nonexistent", (BASE_PATH / "simple",), {}),  # type: ignore
    (
        "test6",
        (BASE_PATH / "simple",),  # type: ignore
        {
            "PROJECT1": {
                "Amon": {
                    "tas": {
                        "cds_var_name": "2m_temperature",
                        "source_var_name": "2t",
                    },
                    "psl": {
                        "cds_var_name": "mean_sea_level_pressure",
                        "source_var_name": "msl",
                    },
                },
            },
        },
    ),
    (
        "test6",
        (BASE_PATH / "simple", BASE_PATH / "override"),  # type: ignore
        {
            "PROJECT1": {
                "Amon": {
                    "tas": {
                        "cds_var_name": "temperature_2m",
                        "source_var_name": "t2m",
                    },
                    "psl": {
                        "cds_var_name": "mean_sea_level_pressure",
                        "source_var_name": "msl",
                    },
                    "uas": {
                        "cds_var_name": "10m_u-component_of_neutral_wind",
                        "source_var_name": "u10n",
                    },
                    "vas": {
                        "cds_var_name": "v-component_of_neutral_wind_at_10m",
                        "source_var_name": "10v",
                    },
                },
            },
        },
    ),
]


@pytest.mark.parametrize(
    ("project", "extra_facets_dir", "expected"),
    TEST_LOAD_EXTRA_FACETS,
)
def test_load_extra_facets(project, extra_facets_dir, expected):
    extra_facets = _load_extra_facets(project, extra_facets_dir)
    assert extra_facets == expected


def test_get_extra_facets(tmp_path):
    dataset = Dataset(
        project="test_project",
        mip="test_mip",
        dataset="test_dataset",
        short_name="test_short_name",
    )
    extra_facets_file = tmp_path / f"{dataset['project']}-test.yml"
    extra_facets_file.write_text(
        textwrap.dedent("""
            {dataset}:
              {mip}:
                {short_name}:
                  key: value
            """)
        .strip()
        .format(**dataset.facets),
    )

    extra_facets = get_extra_facets(dataset, extra_facets_dir=(tmp_path,))

    assert extra_facets == {"key": "value"}


def test_get_extra_facets_cmip3():
    dataset = Dataset(
        project="CMIP3",
        mip="A1",
        short_name="tas",
        dataset="CM3",
    )
    extra_facets = get_extra_facets(dataset, extra_facets_dir=())

    assert extra_facets == {"institute": ["CNRM", "INM", "CNRM_CERFACS"]}


def test_get_extra_facets_cmip5():
    dataset = Dataset(
        project="CMIP5",
        mip="Amon",
        short_name="tas",
        dataset="ACCESS1-0",
    )
    extra_facets = get_extra_facets(dataset, extra_facets_dir=())

    assert extra_facets == {
        "institute": ["CSIRO-BOM"],
        "product": ["output1", "output2"],
    }


def test_get_project_config(mocker):
    mock_result = mocker.Mock()
    mocker.patch.object(_config, "CFG", {"CMIP6": mock_result})

    # Check valid result
    result = _config.get_project_config("CMIP6")
    assert result == mock_result

    # Check error
    with pytest.raises(RecipeError):
        _config.get_project_config("non-existent-project")


def test_load_default_config(cfg_default, monkeypatch):
    """Test that the default configuration can be loaded."""
    project_cfg = {}
    monkeypatch.setattr(_config, "CFG", project_cfg)
    default_dev_file = importlib_files("esmvalcore") / "config-developer.yml"

    session = cfg_default.start_session("recipe_example")

    default_cfg = {
        "auxiliary_data_dir": Path.home() / "auxiliary_data",
        "check_level": CheckLevels.DEFAULT,
        "compress_netcdf": False,
        "config_developer_file": default_dev_file,
        "dask": {
            "profiles": {
                "local_threaded": {
                    "scheduler": "threads",
                },
                "local_distributed": {
                    "cluster": {
                        "type": "distributed.LocalCluster",
                    },
                },
                "debug": {
                    "scheduler": "synchronous",
                },
            },
            "use": "local_threaded",
        },
        "diagnostics": None,
        "download_dir": Path.home() / "climate_data",
        "drs": {
            "CMIP3": "ESGF",
            "CMIP5": "ESGF",
            "CMIP6": "ESGF",
            "CORDEX": "ESGF",
            "obs4MIPs": "ESGF",
        },
        "exit_on_warning": False,
        "extra_facets_dir": [],
        "log_level": "info",
        "logging": {"log_progress_interval": 0.0},
        "max_datasets": None,
        "max_parallel_tasks": None,
        "max_years": None,
        "output_dir": Path.home() / "esmvaltool_output",
        "output_file_type": "png",
        "profile_diagnostic": False,
        "remove_preproc_dir": True,
        "resume_from": [],
        "rootpath": {"default": [Path.home() / "climate_data"]},
        "run_diagnostic": True,
        "search_esgf": "never",
        "skip_nonexistent": False,
        "save_intermediary_cubes": False,
    }

    directory_attrs = {
        "session_dir",
        "plot_dir",
        "preproc_dir",
        "run_dir",
        "work_dir",
        "config_dir",
    }
    # Check that only allowed keys are in it
    assert set(default_cfg) == set(session)

    # Check that all required directories are available
    assert all(hasattr(session, attr) for attr in directory_attrs)

    # Check default values
    for key in default_cfg:
        assert session[key] == default_cfg[key]

    # Check output directories
    assert str(session.session_dir).startswith(
        str(Path.home() / "esmvaltool_output" / "recipe_example"),
    )
    for path in ("preproc", "work", "run"):
        assert getattr(session, path + "_dir") == session.session_dir / path
    assert session.plot_dir == session.session_dir / "plots"
    with pytest.warns(ESMValCoreDeprecationWarning):
        assert session.config_dir is None

    # Check that projects were configured
    assert project_cfg


def test_rootpath_obs4mips_case_correction(monkeypatch):
    """Test that the name of the obs4MIPs project is correct in rootpath."""
    monkeypatch.setitem(CFG, "rootpath", {"obs4mips": "/path/to/data"})
    assert "obs4mips" not in CFG["rootpath"]
    assert CFG["rootpath"]["obs4MIPs"] == [Path("/path/to/data")]


def test_drs_obs4mips_case_correction(monkeypatch):
    """Test that the name of the obs4MIPs project is correct in rootpath."""
    monkeypatch.setitem(CFG, "drs", {"obs4mips": "ESGF"})
    assert "obs4mips" not in CFG["drs"]
    assert CFG["drs"]["obs4MIPs"] == "ESGF"


def test_project_obs4mips_case_correction(tmp_path, monkeypatch, mocker):
    monkeypatch.setattr(_config, "CFG", {})
    mocker.patch.object(_config, "read_cmor_tables", autospec=True)
    cfg_file = tmp_path / "config-developer.yml"
    project_cfg = {"input_dir": {"default": "/"}}
    cfg_dev = {
        "obs4mips": project_cfg,
    }
    with cfg_file.open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg_dev, file)

    _config.load_config_developer(cfg_file)

    assert "obs4mips" not in _config.CFG
    assert _config.CFG["obs4MIPs"] == project_cfg

    # Restore config-developer file
    _config_validators.validate_config_developer(None)


def test_load_config_developer_custom(tmp_path, monkeypatch, mocker):
    monkeypatch.setattr(_config, "CFG", {})
    mocker.patch.object(_config, "read_cmor_tables", autospec=True)
    cfg_file = tmp_path / "config-developer.yml"
    cfg_dev = {"custom": {"cmor_path": "/path/to/tables"}}
    with cfg_file.open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg_dev, file)

    _config.load_config_developer(cfg_file)

    assert "custom" in _config.CFG

    # Restore config-developer file
    _config_validators.validate_config_developer(None)


@pytest.mark.parametrize(
    ("project", "step"),
    [
        ("invalid_project", "load"),
        ("CMIP6", "load"),
        ("EMAC", "save"),
    ],
)
def test_get_ignored_warnings_none(project, step):
    """Test ``get_ignored_warnings``."""
    assert get_ignored_warnings(project, step) is None


def test_get_ignored_warnings_emac():
    """Test ``get_ignored_warnings``."""
    ignored_warnings = get_ignored_warnings("EMAC", "load")
    assert isinstance(ignored_warnings, list)
    assert ignored_warnings
