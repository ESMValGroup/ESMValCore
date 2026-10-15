import re
from importlib.resources import files as importlib_files
from pathlib import Path

import dask.config
import pytest

import esmvalcore.config
from esmvalcore.cmor.check import CheckLevels

BUILTIN_CONFIG_DIR = Path(esmvalcore.config.__file__).parent.joinpath(
    "configurations",
)


@pytest.mark.parametrize(
    "config_file",
    [
        pytest.param(f, id=f.relative_to(BUILTIN_CONFIG_DIR).as_posix())
        for f in BUILTIN_CONFIG_DIR.rglob("*.yml")
    ],
)
def test_builtin_config_files_have_description(config_file: Path) -> None:
    """Test that all built-in config files have a description."""
    # Use the same code to find the description as in the
    # `esmvaltool config list` command.
    first_comment = re.search(
        r"\A((?: *#.*\r?\n)+)",
        config_file.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert first_comment
    description = " ".join(
        line.lstrip(" #").strip()
        for line in first_comment.group(1).split("\n")
    ).strip()
    # Add a basic check that the description is meaningful
    assert len(description) > 15
    assert description.endswith(".")


BASE_PATH = importlib_files("tests") / "sample_data" / "extra_facets"

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


def test_load_default_config(cfg_default, monkeypatch):
    """Test that the default configuration can be loaded."""
    root_path = importlib_files("esmvalcore")
    default_config_dir = root_path / "config" / "configurations" / "defaults"
    default_project_settings = dask.config.collect(
        paths=[str(p) for p in default_config_dir.glob("*.yml")],
        env={},
    )["projects"]

    session = cfg_default.start_session("recipe_example")

    default_cfg = {
        "auxiliary_data_dir": Path.home() / "auxiliary_data",
        "check_level": CheckLevels.DEFAULT,
        "compress_netcdf": False,
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
        "exit_on_warning": False,
        "log_level": "info",
        "logging": {"log_progress_interval": 0.0},
        "max_datasets": None,
        "max_parallel_tasks": None,
        "max_years": None,
        "output_dir": Path.home() / "esmvaltool_output",
        "output_file_type": "png",
        "profile_diagnostic": False,
        "projects": default_project_settings,
        "remove_preproc_dir": True,
        "resume_from": [],
        "run_diagnostic": True,
        "search_data": "quick",
        "skip_nonexistent": False,
        "save_intermediary_cubes": False,
    }

    directory_attrs = {
        "session_dir",
        "plot_dir",
        "preproc_dir",
        "run_dir",
        "work_dir",
    }
    # Check that only allowed keys are in it
    assert set(default_cfg) == set(session)

    # Check that all required directories are available
    assert all(hasattr(session, attr) for attr in directory_attrs)

    # Check default values
    for key, value in default_cfg.items():
        assert session[key] == value

    # Check that project settings were loaded
    assert set(session["projects"]) == {
        # ESGF
        "CMIP3",
        "CMIP5",
        "CMIP6",
        "CMIP6Plus",
        "CMIP7",
        "CORDEX",
        "CORDEX-CMIP6",
        "obs4MIPs",
        "ana4MIPs",
        # ESMValCore supported projects
        "native6",
        "ACCESS",
        "CESM",
        "EMAC",
        "ICON",
        "IPSLCM",
        # ESMValTool CMORizers
        "OBS",
        "OBS6",
    }

    # Check output directories
    assert str(session.session_dir).startswith(
        str(Path.home() / "esmvaltool_output" / "recipe_example"),
    )
    for path in ("preproc", "work", "run"):
        assert getattr(session, path + "_dir") == session.session_dir / path
    assert session.plot_dir == session.session_dir / "plots"
