"""Selection and loading across intake-esm and other data sources."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import intake_esgf
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from esmvalcore.dataset import Dataset

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pytest_mock import MockerFixture

    from esmvalcore.config import Session


FACETS = {
    "activity": "CMIP",
    "institute": "TEST",
    "dataset": "TEST-MODEL",
    "exp": "historical",
    "ensemble": "r1i1p1f1",
    "mip": "Amon",
    "short_name": "tas",
    "grid": "gn",
}
MAPPING = {
    "activity": "activity_id",
    "institute": "institution_id",
    "dataset": "source_id",
    "exp": "experiment_id",
    "ensemble": "member_id",
    "mip": "table_id",
    "short_name": "variable_id",
    "grid": "grid_label",
}


def _write_asset(path: Path, years: tuple[int, int], offset: float) -> None:
    """Write monthly data whose values identify their position in time."""
    path.parent.mkdir(parents=True)
    months = np.arange(12 * (years[0] - 2000), 12 * (years[1] - 1999))
    xr.Dataset(
        {
            "tas": (
                ("time", "lat", "lon"),
                np.broadcast_to(
                    (280.0 + months + offset)[:, None, None],
                    (len(months), 1, 2),
                ).astype("float32"),
                {"standard_name": "air_temperature", "units": "K"},
            ),
        },
        coords={
            "time": (
                "time",
                months * 30 + 15,
                {"units": "days since 2000-01-01", "calendar": "360_day"},
            ),
            "lat": ("lat", [10.0], {"units": "degrees_north"}),
            "lon": ("lon", [0.0, 90.0], {"units": "degrees_east"}),
            "height": (
                (),
                2.0,
                {
                    "standard_name": "height",
                    "units": "m",
                    "positive": "up",
                },
            ),
        },
    ).to_netcdf(path)


def _catalog_config(path: Path, version: str, period: str) -> dict[str, Any]:
    """Describe an asset using the actual intake-esm catalog reader."""
    row = {column: FACETS[facet] for facet, column in MAPPING.items()}
    row.update(version=version, time_range=period, path=str(path))
    catalog_path = path.with_suffix(".json")
    catalog_path.write_text(
        json.dumps(
            {
                "esmcat_version": "0.1.0",
                "id": path.parent.parent.name,
                "description": "Synthetic data for mixed-source selection",
                "catalog_dict": [row],
                "attributes": [
                    {"column_name": key} for key in row if key != "path"
                ],
                "assets": {"column_name": "path", "format": "netcdf"},
                "aggregation_control": {
                    "variable_column_name": "variable_id",
                    "groupby_attrs": list(MAPPING.values()),
                    "aggregations": [],
                },
            },
        ),
    )
    return {
        "type": "esmvalcore.io.intake_esm.IntakeEsmDataSource",
        "catalog": str(catalog_path),
        "facets": {**MAPPING, "version": "version", "timerange": "time_range"},
        "to_dask_kwargs": {"xarray_open_kwargs": {"decode_cf": False}},
    }


@pytest.fixture
def mixed_sources(
    tmp_path: Path,
    session: Session,
    mocker: MockerFixture,
) -> Callable[..., tuple[Dataset, dict[str, Path]]]:
    """Configure real sources, replacing only ESGF network operations."""
    mocker.patch.dict(
        intake_esgf.conf,
        {
            "local_cache": [str(tmp_path / "cache")],
            "download_db": str(tmp_path / "downloads.db"),
            "logfile": str(tmp_path / "esgf.log"),
            "esg_dataroot": [],
        },
    )

    def configure(
        other_kind: str,
        *,
        catalog_version: str = "v20200101",
        other_version: str = "v20200101",
        catalog_years: tuple[int, int] = (2000, 2001),
        other_years: tuple[int, int] = (2000, 2001),
        catalog_priority: int = 1,
        search_data: str = "complete",
    ) -> tuple[Dataset, dict[str, Path]]:
        paths = {}
        configs = {}
        for name, years, version in (
            ("catalog", catalog_years, catalog_version),
            ("other", other_years, other_version),
        ):
            period = f"{years[0]}01-{years[1]}12"
            paths[name] = (
                tmp_path
                / name
                / version
                / (f"tas_Amon_TEST-MODEL_historical_r1i1p1f1_gn_{period}.nc")
            )
            _write_asset(paths[name], years, 100.0 if name == "other" else 0.0)
            configs[name] = _catalog_config(paths[name], version, period)

        if other_kind == "local":
            configs["other"] = {
                "type": "esmvalcore.io.local.LocalDataSource",
                "rootpath": str(tmp_path / "other"),
                "dirname_template": "{version}",
                "filename_template": "{short_name}_{mip}_{dataset}_{exp}_{ensemble}_{grid}_*.nc",
            }
        elif other_kind == "esgf":
            configs["other"] = {
                "type": "esmvalcore.io.intake_esgf.IntakeESGFDataSource",
                "facets": {**MAPPING, "project": "project"},
            }

            def search(cat: intake_esgf.ESGFCatalog, **kwargs: Any) -> None:
                cat.last_search = kwargs
                cat.df = pd.DataFrame(
                    [
                        {
                            **{
                                column: FACETS[facet]
                                for facet, column in MAPPING.items()
                            },
                            "project": "CMIP6",
                            "mip_era": "CMIP6",
                            "version": other_version.removeprefix("v"),
                        },
                    ],
                )

            def to_paths(
                cat: intake_esgf.ESGFCatalog,
                **_kwargs: Any,
            ) -> dict[str, list[Path]]:
                return {cat.df["key"].iloc[0]: [paths["other"]]}

            def to_datasets(
                cat: intake_esgf.ESGFCatalog,
                **_kwargs: Any,
            ) -> dict[str, xr.Dataset]:
                with xr.open_dataset(paths["other"], decode_cf=False) as data:
                    return {cat.df["key"].iloc[0]: data.load()}

            for method, replacement in (
                ("search", search),
                ("to_path_dict", to_paths),
                ("to_dataset_dict", to_datasets),
            ):
                mocker.patch.object(
                    intake_esgf.ESGFCatalog,
                    method,
                    autospec=True,
                    side_effect=replacement,
                )

        configs["catalog"]["priority"] = catalog_priority
        configs["other"]["priority"] = 3 - catalog_priority
        session["projects"]["CMIP6"]["data"] = configs
        session["search_data"] = search_data
        dataset = Dataset(
            project="CMIP6",
            **FACETS,
            frequency="mon",
            timerange="200001/200112",
        )
        dataset.session = session
        return dataset, paths

    return configure


def _assert_selected_asset(dataset: Dataset, expected: Path) -> None:
    """Check the values identify the preferred source, without duplication."""
    cube = dataset.load()
    offset = 100.0 if expected.parent.parent.name == "other" else 0.0
    np.testing.assert_array_equal(
        cube.data[:, 0, 0],
        280.0 + np.arange(24) + offset,
    )


@pytest.mark.parametrize("catalog_priority", [1, 2])
@pytest.mark.parametrize(
    ("search_data", "other_kind"),
    [
        ("quick", "local"),
        ("quick", "esgf"),
        ("quick", "intake"),
        ("complete", "local"),
        ("complete", "esgf"),
        ("complete", "intake"),
    ],
)
def test_duplicate_sources_use_priority(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
    other_kind: str,
    catalog_priority: int,
    search_data: str,
) -> None:
    """The same version and period should be loaded once from the best source."""
    dataset, paths = mixed_sources(
        other_kind,
        catalog_priority=catalog_priority,
        search_data=search_data,
    )
    winner = "catalog" if catalog_priority == 1 else "other"
    _assert_selected_asset(dataset, paths[winner])


@pytest.mark.parametrize(
    "other_kind",
    [
        "local",
        "esgf",
        "intake",
    ],
)
@pytest.mark.parametrize(
    ("catalog_version", "other_version", "winner"),
    [
        ("v20200101", "v20210101", "other"),
        ("v20210101", "v20200101", "catalog"),
        ("v9", "v10", "other"),
    ],
)
def test_complete_search_uses_newest_version(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
    other_kind: str,
    catalog_version: str,
    other_version: str,
    winner: str,
) -> None:
    """Complete search should prefer the newest version across sources."""
    dataset, paths = mixed_sources(
        other_kind,
        catalog_version=catalog_version,
        other_version=other_version,
    )
    _assert_selected_asset(dataset, paths[winner])


def test_catalog_versions_with_different_prefixes(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
) -> None:
    """A missing v prefix should not make a newer catalog version older."""
    dataset, paths = mixed_sources(
        "intake",
        catalog_version="20210101",
        other_version="v20200101",
    )
    _assert_selected_asset(dataset, paths["catalog"])


@pytest.mark.parametrize(
    ("search_data", "other_kind"),
    [
        ("quick", "local"),
        ("quick", "esgf"),
        ("quick", "intake"),
        ("complete", "local"),
        ("complete", "esgf"),
        ("complete", "intake"),
    ],
)
def test_complementary_sources_preserve_time_coverage(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
    other_kind: str,
    search_data: str,
) -> None:
    """Two sources covering different years must both contribute their data."""
    dataset, _ = mixed_sources(
        other_kind,
        catalog_years=(2000, 2000),
        other_years=(2001, 2001),
        search_data=search_data,
    )
    dataset.find_files()
    assert len(dataset.files) == 2
    cube = dataset.load()
    np.testing.assert_array_equal(
        cube.data[:, 0, 0],
        280.0 + np.arange(24) + np.repeat([0.0, 100.0], 12),
    )
    time = cube.coord("time")
    dates = time.units.num2date(time.points)
    assert [(date.year, date.month) for date in dates] == [
        (year, month) for year in (2000, 2001) for month in range(1, 13)
    ]


@pytest.mark.parametrize(
    "other_kind",
    [
        "local",
        "esgf",
        "intake",
    ],
)
def test_lower_priority_subset_does_not_duplicate_months(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
    other_kind: str,
) -> None:
    """A lower-priority copy of one year must not duplicate that year's data."""
    dataset, paths = mixed_sources(other_kind, other_years=(2001, 2001))
    _assert_selected_asset(dataset, paths["catalog"])
    cube = dataset.load()
    np.testing.assert_array_equal(cube.data[:, 0, 0], 280.0 + np.arange(24))


@pytest.mark.parametrize("other_kind", ["local", "esgf", "intake"])
@pytest.mark.parametrize("search_data", ["quick", "complete"])
def test_newer_partial_version_preserves_older_coverage(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
    other_kind: str,
    search_data: str,
) -> None:
    """A newer release replaces overlapping samples without losing a year."""
    dataset, _ = mixed_sources(
        other_kind,
        other_version="v20210101",
        other_years=(2001, 2001),
        search_data=search_data,
    )
    cube = dataset.load()
    offsets = np.repeat([0.0, 100.0 if search_data == "complete" else 0.0], 12)
    np.testing.assert_array_equal(
        cube.data[:, 0, 0],
        280.0 + np.arange(24) + offsets,
    )


@pytest.mark.parametrize("other_kind", ["local", "esgf", "intake"])
@pytest.mark.parametrize("months", [list(range(6)), [0, 2, 4, 6, 8, 10]])
def test_quick_search_fills_missing_months_in_preferred_source(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
    other_kind: str,
    months: list[int],
) -> None:
    """Coverage of a year does not imply that every month is available."""
    dataset, paths = mixed_sources(
        other_kind,
        other_years=(2000, 2000),
        catalog_priority=2,
        search_data="quick",
    )
    dataset["timerange"] = "200001/200012"
    with xr.open_dataset(paths["other"], decode_cf=False) as data:
        subset = data.isel(time=months).load()
    subset.to_netcdf(paths["other"])
    cube = dataset.load()
    expected = 280.0 + np.arange(12)
    expected[months] += 100.0
    np.testing.assert_array_equal(cube.data[:, 0, 0], expected)
