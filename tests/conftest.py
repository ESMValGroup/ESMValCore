from __future__ import annotations

import pickle
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING

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
from woodpecker.fixes.labels import Labels
from woodpecker.fixes.registry import FixFunction, register_fix_function

import esmvalcore.cmor.table
import esmvalcore.io._woodpecker
from esmvalcore.config import CFG, Config

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

    from esmvalcore.config import Session


@lru_cache
def _load_default_config() -> bytes:
    """Create a pickled configuration object with default values."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Do not instantiate `Config` objects directly",
            category=UserWarning,
        )
        cfg = Config()
    cfg.load_from_dirs([])
    return pickle.dumps(cfg)


@pytest.fixture
def cfg_default():
    """Create a configuration object with default values."""
    # Unpickling is much faster than copy.deepcopy and the pickled data
    # is created by this process, so it is safe to load.
    return pickle.loads(_load_default_config())  # noqa: S301


@pytest.fixture(autouse=True)
def ignore_existing_user_config(
    monkeypatch: pytest.MonkeyPatch,
    cfg_default: Config,
) -> None:
    """Ignore user's configuration when running tests."""
    monkeypatch.setattr(CFG, "_mapping", cfg_default._mapping)
    monkeypatch.setattr(
        esmvalcore.cmor.table,
        "CMOR_TABLES",
        {
            project: esmvalcore.cmor.table.get_tables(
                session=cfg_default,
                project=project,
            )
            for project in cfg_default["projects"]
        },
    )


@pytest.fixture
def session(
    tmp_path: Path,
    ignore_existing_user_config: None,
    monkeypatch: pytest.MonkeyPatch,
) -> Session:
    """Session object with default settings."""
    monkeypatch.setitem(CFG, "output_dir", tmp_path / "esmvaltool_output")
    return CFG.start_session("recipe_test")


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


@register_fix_function
class SetFixedBy(FixFunction):
    """Woodpecker fix used by the tests: set the global attribute ``fixed_by``."""

    prefix = "esmvaltest"
    suffix = "set_fixed_by"
    name = "Set fixed_by"
    description = "Sets the global attribute fixed_by."
    categories = ["metadata"]  # noqa: RUF012
    priority = 10
    labels = [Labels.RISK_REVIEW_BEFORE_APPLYING]  # noqa: RUF012

    def matches(self, dataset: xr.Dataset) -> bool:
        return dataset.attrs.get("fixed_by") != "woodpecker"

    def apply(self, dataset: xr.Dataset, dry_run: bool = True) -> bool:
        changed = self.matches(dataset)
        if changed and not dry_run:
            dataset.attrs["fixed_by"] = "woodpecker"
        return changed


WOODPECKER_RECIPES = """\
recipes:
  - id: esmvaltest.fix_test_data
    description: Fix the test data, matched on the path
    match:
      path_patterns:
        - "*TEST-DATA*"
    steps:
      - id: esmvaltest.set_fixed_by
  - id: esmvaltest.fix_test_model
    description: Fix the test data, matched on the attributes
    match:
      attrs:
        source_id: TEST-MODEL
    steps:
      - id: esmvaltest.set_fixed_by
"""
"""Woodpecker recipes used by the tests.

The recipe ``esmvaltest.fix_test_data`` matches data whose name contains
``TEST-DATA``, the recipe ``esmvaltest.fix_test_model`` matches data with the
global attribute ``source_id: TEST-MODEL``.
"""


@pytest.fixture
def woodpecker_recipe_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Make woodpecker use the recipes in :data:`WOODPECKER_RECIPES`."""
    path = tmp_path / "woodpecker_recipes"
    path.mkdir()
    (path / "esmvaltest.yaml").write_text(WOODPECKER_RECIPES, encoding="utf-8")
    monkeypatch.setenv("WOODPECKER_RECIPE_PATH", str(path))
    # The recipe catalog is cached per value of WOODPECKER_RECIPE_PATH.
    monkeypatch.setattr(esmvalcore.io._woodpecker, "_RECIPE_CATALOGS", {})
    return path
