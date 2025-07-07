import warnings
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from cf_units import Unit
from iris.coords import (
    AncillaryVariable,
    AuxCoord,
    CellMeasure,
    CellMethod,
    DimCoord,
)
from iris.cube import Cube

import esmvalcore.config._dask
from esmvalcore.config import CFG, Config


@lru_cache
def _load_default_config():
    """Create a configuration object with default values."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Do not instantiate `Config` objects directly",
            category=UserWarning,
            module="esmvalcore",
        )
        cfg = Config()
    cfg.load_from_dirs([])
    return cfg


@pytest.fixture
def cfg_default():
    """Create a configuration object with default values."""
    cfg = _load_default_config()
    return deepcopy(cfg)


@pytest.fixture(autouse=True)
def ignore_existing_user_config(monkeypatch, cfg_default):
    """Ignore user's configuration when running tests."""
    monkeypatch.setattr(CFG, "_mapping", cfg_default._mapping)


@pytest.fixture
def session(tmp_path: Path, ignore_existing_user_config, monkeypatch):
    """Session object with default settings."""
    monkeypatch.setitem(CFG, "rootpath", {"default": {tmp_path: "default"}})
    monkeypatch.setitem(CFG, "output_dir", tmp_path / "esmvaltool_output")
    return CFG.start_session("recipe_test")


# TODO: remove in v2.14.0
@pytest.fixture(autouse=True)
def ignore_old_config_user(tmp_path, monkeypatch):
    """Ignore potentially existing old config-user.yml file in all tests."""
    nonexistent_config_dir = tmp_path / "nonexistent_config_dir"
    monkeypatch.setattr(
        Config,
        "_DEFAULT_USER_CONFIG_DIR",
        nonexistent_config_dir,
    )


# TODO: remove in v2.14.0
@pytest.fixture(autouse=True)
def ignore_old_dask_config_file(tmp_path, monkeypatch):
    """Ignore potentially existing old dask.yml file in all tests."""
    nonexistent_file = tmp_path / "nonexistent_file.yml"
    monkeypatch.setattr(
        esmvalcore.config._dask,
        "CONFIG_FILE",
        nonexistent_file,
    )


@pytest.fixture
def realistic_4d_cube():
    """Create a realistic 4D cube."""
    time = DimCoord(
        [11.0, 12.0],
        standard_name="time",
        units=Unit("hours since 1851-01-01", calendar="360_day"),
    )
    plev = DimCoord([50000], standard_name="air_pressure", units="Pa")
    lat = DimCoord([0.0, 1.0], standard_name="latitude", units="degrees_north")
    lon = DimCoord(
        [0.0, 20.0, 345.0],
        standard_name="longitude",
        units="degrees_east",
    )

    aux_2d_data = np.arange(2 * 3).reshape(2, 3)
    aux_2d_bounds = np.stack(
        (aux_2d_data - 1, aux_2d_data, aux_2d_data + 1),
        axis=-1,
    )
    aux_2d = AuxCoord(aux_2d_data, var_name="aux_2d")
    aux_2d_with_bnds = AuxCoord(
        aux_2d_data,
        bounds=aux_2d_bounds,
        var_name="aux_2d_with_bnds",
    )
    aux_time = AuxCoord(["Jan", "Jan"], var_name="aux_time")
    aux_lon = AuxCoord([0, 1, 2], var_name="aux_lon")

    cell_area = CellMeasure(
        np.arange(2 * 2 * 3).reshape(2, 2, 3) + 10,
        standard_name="cell_area",
        units="m2",
        measure="area",
    )
    type_var = AncillaryVariable(
        [["sea", "land", "lake"], ["lake", "sea", "land"]],
        var_name="type",
        units="no_unit",
    )

    return Cube(
        np.ma.masked_inside(
            np.arange(2 * 1 * 2 * 3).reshape(2, 1, 2, 3),
            1,
            3,
        ),
        var_name="ta",
        standard_name="air_temperature",
        long_name="Air Temperature",
        units="K",
        cell_methods=[CellMethod("mean", "time")],
        dim_coords_and_dims=[(time, 0), (plev, 1), (lat, 2), (lon, 3)],
        aux_coords_and_dims=[
            (aux_2d, (0, 3)),
            (aux_2d_with_bnds, (0, 3)),
            (aux_time, 0),
            (aux_lon, 3),
        ],
        cell_measures_and_dims=[(cell_area, (0, 2, 3))],
        ancillary_variables_and_dims=[(type_var, (0, 3))],
        attributes={"test": 1},
    )
