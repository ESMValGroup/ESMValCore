"""Tests for the basic configuration of the preprocessor module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import esmvalcore
from esmvalcore.config import CFG
from esmvalcore.dataset import Dataset
from esmvalcore.exceptions import RecipeError
from esmvalcore.preprocessor import (
    DEFAULT_ORDER,
    FINAL_STEPS,
    INITIAL_STEPS,
    MULTI_MODEL_FUNCTIONS,
    TIME_PREPROCESSORS,
    _get_preprocessor_filename,
)

if TYPE_CHECKING:
    from esmvalcore.config import Session
    from esmvalcore.typing import Facets


def test_non_repeated_keys():
    """Check that there are not repeated keys in the lists."""
    assert len(DEFAULT_ORDER) == len(set(DEFAULT_ORDER))
    assert len(TIME_PREPROCESSORS) == len(set(TIME_PREPROCESSORS))
    assert len(INITIAL_STEPS) == len(set(INITIAL_STEPS))
    assert len(FINAL_STEPS) == len(set(FINAL_STEPS))
    assert len(MULTI_MODEL_FUNCTIONS) == len(set(MULTI_MODEL_FUNCTIONS))


def test_time_preprocessores_default_order_added():
    assert all(
        time_preproc in DEFAULT_ORDER for time_preproc in TIME_PREPROCESSORS
    )


def test_multimodel_functions_in_default_order():
    assert all(
        time_preproc in DEFAULT_ORDER for time_preproc in MULTI_MODEL_FUNCTIONS
    )


@pytest.mark.parametrize(
    ("facets", "filename"),
    [
        (
            {
                "project": "CMIP6",
                "mip": "Amon",
                "short_name": "tas",
                "dataset": "GFDL-ESM4",
                "ensemble": "r1i1p1f1",
                "exp": ["historical", "ssp585"],
                "version": "v20191115",
                "grid": "gn",
                "timerange": "1850/P250Y",
            },
            "CMIP6_GFDL-ESM4_Amon_historical-ssp585_r1i1p1f1_tas_gn_18500101-21000101.nc",
        ),
        (
            {
                "project": "CMIP6",
                "mip": "fx",
                "short_name": "areacella",
                "dataset": "GFDL-ESM4",
                "ensemble": "r1i1p1f1",
                "exp": "historical",
                "version": "v20191115",
                "grid": "gn",
            },
            "CMIP6_GFDL-ESM4_fx_historical_r1i1p1f1_areacella_gn.nc",
        ),
    ],
)
def test_get_preprocessor_filename(
    session: Session,
    facets: Facets,
    filename: str,
) -> None:
    """Test the function `_get_preprocessor_filename`."""
    dataset = Dataset(**facets)
    dataset.session = session
    result = _get_preprocessor_filename(dataset)
    expected = session.preproc_dir / filename
    assert result == expected


def test_get_preprocessor_filename_missing_facet(session: Session) -> None:
    dataset = Dataset(
        project="CMIP6",
        mip="Amon",
        short_name="tas",
    )
    dataset.session = session
    with pytest.raises(
        RecipeError,
        match=r"missing facets: 'dataset', 'ensemble', 'exp', 'grid'.",
    ):
        _get_preprocessor_filename(dataset)


def test_get_preprocessor_filename_wrong_type_facet(session: Session) -> None:
    dataset = Dataset(
        project="CMIP6",
        mip="Amon",
        short_name="tas",
        dataset="GFDL-ESM4",
        ensemble="r1i1p1f1",
        exp={"historical", "ssp585"},  # type: ignore[arg-type]
        grid="gn",
    )
    dataset.session = session
    with pytest.raises(
        RecipeError,
        match=r".* facet values have invalid type: 'exp: .*' has type set",
    ):
        _get_preprocessor_filename(dataset)


def test_get_preprocessor_filename_default(
    session: Session,
) -> None:
    """Test the function `_get_preprocessor_filename`."""
    session["projects"]["CMIP6"].pop("preprocessor_filename_template")
    dataset = Dataset(
        project="TestProject",
        mip="Amon",
        short_name="tas",
        dataset="TestModel",
        version="v20191115",
        grid="gn",
        timerange="1850/2100",
        # Add some facets of the wrong type that are not used in the filename.
        ignore=[1, 2],  # type: ignore[list-item]
        ignore_too={"a": 1},  # type: ignore[arg-type]
    )
    # Add a facet that is not used in the filename.
    dataset.facets["long_name"] = "Surface Air Temperature"
    dataset.session = session
    result = _get_preprocessor_filename(dataset)
    filename = "TestModel_gn_Amon_TestProject_tas_v20191115_1850-2100.nc"
    expected = session.preproc_dir / filename
    assert result == expected


def test_get_preprocessor_filename_falls_back_to_config_developer(
    monkeypatch: pytest.MonkeyPatch,
    session: Session,
) -> None:
    """Test the function `_get_preprocessor_filename`."""
    monkeypatch.setitem(
        CFG,
        "config_developer_file",
        Path(esmvalcore.__path__[0], "config-developer.yml"),
    )
    session["projects"]["CMIP6"].pop("preprocessor_filename_template")
    dataset = Dataset(
        project="CMIP6",
        mip="Amon",
        short_name="tas",
        dataset="GFDL-ESM4",
        ensemble="r1i1p1f1",
        exp=["historical", "ssp585"],
        version="v20191115",
        grid="gn",
        timerange="1850/2100",
    )
    dataset.session = session
    result = _get_preprocessor_filename(dataset)
    filename = (
        "CMIP6_GFDL-ESM4_Amon_historical-ssp585_r1i1p1f1_tas_gn_1850-2100.nc"
    )
    expected = session.preproc_dir / filename
    assert result == expected
