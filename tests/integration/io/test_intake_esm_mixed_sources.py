"""Selection and loading across intake-esm and other data sources."""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import intake_esgf
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from prov.model import ProvDocument

from esmvalcore._recipe import check
from esmvalcore._recipe.recipe import read_recipe_file
from esmvalcore.cmor.check import CheckLevels, CMORCheckError
from esmvalcore.dataset import Dataset
from esmvalcore.exceptions import InputFilesNotFound
from esmvalcore.io.intake_esm import IntakeEsmDataset

if TYPE_CHECKING:
    from collections.abc import Callable

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


@pytest.mark.parametrize(
    ("experiment", "activity"),
    [("historical", "CMIP"), ("ssp126", "ScenarioMIP")],
)
@pytest.mark.parametrize("selection", ["explicit", "inferred", "discovered"])
def test_nci_example_preserves_activity(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
    experiment: str,
    activity: str,
    selection: str,
) -> None:
    """NCI project_id must not override activity when mixing catalog sources."""
    dataset, paths = mixed_sources("intake")
    sources = dataset.session["projects"]["CMIP6"]["data"]
    config_path = (
        importlib.resources.files("esmvalcore.config")
        / "configurations"
        / "data-intake-esm-nci.yml"
    )
    config = yaml.safe_load(config_path.read_text())
    nci_source = config["projects"]["CMIP6"]["data"]["nci-disk-catalog"]
    nci_source.update(
        catalog=sources["catalog"]["catalog"],
        priority=sources["catalog"]["priority"],
    )
    for name, path in paths.items():
        catalog_path = path.with_suffix(".json")
        catalog = json.loads(catalog_path.read_text())
        row = catalog["catalog_dict"][0]
        row["experiment_id"] = experiment
        row["activity_id"] = activity
        if name == "catalog":
            # Real fs38 rows use CMIP even for ScenarioMIP assets.
            row.pop("activity_id")
            row.update(project_id="CMIP", file_type="l")
            catalog["aggregation_control"]["groupby_attrs"] = [
                "project_id" if column == "activity_id" else column
                for column in catalog["aggregation_control"]["groupby_attrs"]
            ]
            catalog["attributes"] = [
                {"column_name": key} for key in row if key != "path"
            ]
        catalog_path.write_text(json.dumps(catalog))
    sources["catalog"] = nci_source
    dataset["exp"] = experiment
    if selection == "explicit":
        dataset["activity"] = activity
    else:
        dataset.facets.pop("activity")
        if selection == "discovered":
            dataset["exp"] = "*"
            discovered = list(dataset.from_files())
            assert len(discovered) == 1
            dataset = discovered[0]
    dataset.find_files()
    assert dataset["exp"] == experiment
    assert dataset["activity"] == activity
    nci_element, other_element = dataset.files
    assert "activity" not in nci_element.facets
    assert other_element.facets["activity"] == activity
    _assert_selected_asset(dataset, paths["catalog"])


def test_recipe_reuses_catalog_across_preprocessing_tasks(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
    recipe_documentation: str,
    tmp_path: Path,
) -> None:
    """Run two profiles sharing each of two distinct catalog selections."""
    dataset, _ = mixed_sources(
        "intake",
        catalog_years=(2000, 2000),
        other_years=(2001, 2001),
    )
    session = dataset.session
    session["max_parallel_tasks"] = 1
    session.run_dir.mkdir(parents=True)
    content = {
        "datasets": [
            {"project": "CMIP6", **FACETS, "timerange": "200001/200112"},
        ],
        "preprocessors": {
            "monthly": {},
            "annual": {"annual_statistics": {"operator": "mean"}},
        },
        "diagnostics": {
            "temperature": {
                "variables": {
                    name: {"short_name": "tas", "preprocessor": name}
                    for name in ("monthly", "annual")
                },
                "scripts": None,
            },
        },
    }
    path = tmp_path / "recipe_catalog.yml"
    path.write_text(recipe_documentation + yaml.safe_dump(content))
    recipe = read_recipe_file(path, session)
    assert len(recipe._download_files) == 2
    recipe.run()
    expected = 280.0 + np.arange(24) + np.repeat([0.0, 100.0], 12)
    for profile, values, years in (
        ("monthly", expected, np.repeat([2000, 2001], 12)),
        ("annual", expected.reshape(2, 12).mean(axis=1), [2000, 2001]),
    ):
        outputs = list(
            (session.preproc_dir / "temperature" / profile).glob("*.nc"),
        )
        assert len(outputs) == 1
        with xr.open_dataset(outputs[0]) as output:
            np.testing.assert_array_equal(
                output.tas.to_numpy()[:, 0, 0],
                values,
            )
            np.testing.assert_array_equal(output.time.dt.year, years)
            assert len(np.unique(output.time.to_numpy())) == len(values)
        provenance = ProvDocument.deserialize(
            outputs[0].with_name(f"{outputs[0].stem}_provenance.xml"),
            format="xml",
        )
        assert any(
            str(name) == "attribute:intake_esm_attrs:_data_format_"
            and value == "netcdf"
            for record in provenance.get_records()
            for name, value in record.attributes
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


def test_complementary_catalogs_with_different_coordinate_history(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
) -> None:
    """CMOR coordinate history must not prevent combining available years."""
    dataset, paths = mixed_sources(
        "intake",
        catalog_years=(2000, 2000),
        other_years=(2001, 2001),
    )
    sources = dataset.session["projects"]["CMIP6"]["data"]
    for name, path in paths.items():
        with xr.open_dataset(path, decode_cf=False) as original:
            annotated = original.load()
        annotated["lat"].attrs["history"] = f"CMOR processing for {name}"
        annotated_path = path.with_stem(f"{path.stem}_history")
        annotated.to_netcdf(annotated_path)
        period = "200001-200012" if name == "catalog" else "200101-200112"
        sources[name] = _catalog_config(annotated_path, "v20200101", period)
    cube = dataset.load()
    assert cube.has_lazy_data()
    np.testing.assert_array_equal(
        cube.data[:, 0, 0],
        280.0 + np.arange(24) + np.repeat([0.0, 100.0], 12),
    )
    dates = cube.coord("time").units.num2date(cube.coord("time").points)
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


@pytest.mark.parametrize("known_coverage", [False, True])
@pytest.mark.parametrize("available_year", [2000, 2001])
def test_catalog_rejects_missing_requested_year(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
    known_coverage: bool,
    available_year: int,
) -> None:
    """Missing leading or trailing coverage must not silently shorten output."""
    dataset, _ = mixed_sources(
        "intake",
        catalog_years=(available_year, available_year),
        other_years=(available_year, available_year),
    )
    if known_coverage:
        with pytest.raises(
            InputFilesNotFound,
            match="No input data available",
        ):
            check.data_availability(dataset)
    else:
        for source in dataset.session["projects"]["CMIP6"]["data"].values():
            source["facets"].pop("timerange")
        check.data_availability(dataset)
    with pytest.raises(InputFilesNotFound, match="do not cover requested"):
        dataset.load()


def test_recipe_rejects_incomplete_unknown_catalog_coverage(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
    recipe_documentation: str,
    tmp_path: Path,
) -> None:
    """A recipe cannot label one year of data as a requested two-year period."""
    dataset, _ = mixed_sources(
        "intake",
        catalog_years=(2000, 2000),
        other_years=(2000, 2000),
    )
    session = dataset.session
    for source in session["projects"]["CMIP6"]["data"].values():
        source["facets"].pop("timerange")
    session["max_parallel_tasks"] = 1
    session.run_dir.mkdir(parents=True)
    path = tmp_path / "recipe_incomplete.yml"
    path.write_text(
        recipe_documentation
        + yaml.safe_dump(
            {
                "datasets": [
                    {
                        "project": "CMIP6",
                        **FACETS,
                        "timerange": "200001/200112",
                    },
                ],
                "diagnostics": {
                    "temperature": {"variables": {"tas": {}}, "scripts": None},
                },
            },
        ),
    )
    recipe = read_recipe_file(path, session)
    with pytest.raises(InputFilesNotFound, match="do not cover requested"):
        recipe.run()
    assert not list(session.preproc_dir.rglob("*.nc"))


@pytest.mark.parametrize("unknown_coverage", ["unmapped", "null"])
def test_unknown_catalog_coverage_defers_missing_year_check(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
    unknown_coverage: str,
) -> None:
    """Unknown metadata must not hide a usable complementary source."""
    dataset, _ = mixed_sources(
        "intake",
        catalog_years=(2000, 2000),
        other_years=(2001, 2001),
    )
    other = dataset.session["projects"]["CMIP6"]["data"]["other"]
    if unknown_coverage == "unmapped":
        other["facets"].pop("timerange")
    else:
        catalog = Path(other["catalog"])
        metadata = json.loads(catalog.read_text())
        metadata["catalog_dict"][0]["time_range"] = None
        catalog.write_text(json.dumps(metadata))
    check.data_availability(dataset)
    cube = dataset.load()
    assert cube.shape[0] == 24


def test_catalog_metadata_does_not_fill_gaps_between_assets(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
) -> None:
    """Separated catalog intervals must not imply that the gap is covered."""
    dataset, _ = mixed_sources(
        "intake",
        catalog_years=(2000, 2000),
        other_years=(2002, 2002),
    )
    sources = dataset.session["projects"]["CMIP6"]["data"]
    catalog = Path(sources["catalog"]["catalog"])
    metadata = json.loads(catalog.read_text())
    metadata["catalog_dict"].extend(
        json.loads(Path(sources["other"]["catalog"]).read_text())[
            "catalog_dict"
        ],
    )
    catalog.write_text(json.dumps(metadata))
    sources.pop("other")
    dataset["timerange"] = "2000/2002"
    with pytest.raises(InputFilesNotFound, match="years 2001"):
        check.data_availability(dataset)
    assert len(dataset.files) == 1
    element = dataset.files[0]
    assert isinstance(element, IntakeEsmDataset)
    assert element.timeranges == (
        "200001/200012",
        "200201/200212",
    )


@pytest.mark.parametrize("missing_month", [0, 12, 23])
def test_catalog_coverage_is_checked_against_actual_samples(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
    missing_month: int,
) -> None:
    """Metadata claiming full coverage cannot hide missing monthly samples."""
    dataset, paths = mixed_sources("intake")
    for path in paths.values():
        with xr.open_dataset(path, decode_cf=False) as data:
            subset = data.isel(
                time=np.delete(np.arange(24), missing_month),
            ).load()
        subset.to_netcdf(path)
    check.data_availability(dataset)
    error = CMORCheckError if missing_month == 12 else InputFilesNotFound
    message = "Frequency" if missing_month == 12 else "do not cover requested"
    with pytest.raises(error, match=message):
        dataset.load()


def test_catalog_metadata_overstates_actual_coverage(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
) -> None:
    """A claimed second year must be checked against the actual data."""
    dataset, paths = mixed_sources("intake")
    for path in paths.values():
        with xr.open_dataset(path, decode_cf=False) as data:
            subset = data.isel(time=slice(0, 12)).load()
        subset.to_netcdf(path)
    check.data_availability(dataset)
    with pytest.raises(InputFilesNotFound, match="do not cover requested"):
        dataset.load()


def test_unknown_catalog_missing_internal_year(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
) -> None:
    """Missing years remain errors even when CMOR cadence checks are disabled."""
    dataset, paths = mixed_sources(
        "intake",
        catalog_years=(2000, 2002),
        other_years=(2000, 2002),
    )
    dataset["timerange"] = "2000/2002"
    dataset.session["check_level"] = CheckLevels.IGNORE
    for source in dataset.session["projects"]["CMIP6"]["data"].values():
        source["facets"].pop("timerange")
    for path in paths.values():
        with xr.open_dataset(path, decode_cf=False) as data:
            subset = data.isel(
                time=list(range(12)) + list(range(24, 36)),
            ).load()
        subset.to_netcdf(path)
    check.data_availability(dataset)
    with pytest.raises(InputFilesNotFound, match="do not cover requested"):
        dataset.load()


def test_unknown_local_coverage_defers_catalog_year_check(
    mixed_sources: Callable[..., tuple[Dataset, dict[str, Path]]],
) -> None:
    """An alternative source with unknown intervals may complete a catalog."""
    dataset, _ = mixed_sources(
        "local",
        catalog_years=(2000, 2000),
        other_years=(2001, 2001),
    )
    dataset.find_files()
    for element in dataset.files:
        if not isinstance(element, IntakeEsmDataset):
            element.facets.pop("timerange")
    check.data_availability(dataset)
    assert dataset.load().shape[0] == 24


@pytest.mark.parametrize(
    ("frequency", "samples", "timerange", "error"),
    [
        ("subhrPt", 48, "20000101T000000/20000101T235959", None),
        ("subhrPt", 47, "20000101T000000/20000101T235959", "do not cover"),
        ("subhrPt", 4, "2000/2001", "do not cover"),
        ("subhrPt", 1, "20000101T000000/20000101T235959", "Cannot validate"),
        ("dec", 2, "2000/2019", None),
        ("dec", 1, "2000/2019", "do not cover"),
    ],
)
def test_catalog_subhourly_and_decadal_coverage(
    tmp_path: Path,
    session: Session,
    frequency: str,
    samples: int,
    timerange: str,
    error: str | None,
) -> None:
    """Validate supported frequencies through the actual catalog/Dataset path."""
    subhourly = frequency == "subhrPt"
    points = (
        np.arange(samples) / 48
        if subhourly
        else 1800 + 3600 * np.arange(samples)
    )
    coords: dict[str, Any] = {
        "time": (
            "time",
            points,
            {
                "standard_name": "time",
                "units": "days since 2000-01-01",
                "calendar": "360_day",
            },
        ),
    }
    shape: tuple[int, ...]
    dims: tuple[str, ...]
    if subhourly:
        mip, short_name, name, units = "Esubhr", "tas", "air_temperature", "K"
        shape = (samples, 2, 2)
        dims = ("time", "lat", "lon")
        coords.update(
            lat=(
                "lat",
                [-10.0, 10.0],
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            lon=(
                "lon",
                [0.0, 90.0],
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            height=(
                (),
                2.0,
                {"standard_name": "height", "units": "m", "positive": "up"},
            ),
        )
    else:
        mip, short_name, name, units = (
            "Odec",
            "bigthetaoga",
            "sea_water_conservative_temperature",
            "degC",
        )
        shape = (samples,)
        dims = ("time",)
        coords["time"][2]["bounds"] = "time_bnds"
    variables: dict[str, Any] = {
        short_name: (
            dims,
            np.full(shape, 10.0, dtype="float32"),
            {"standard_name": name, "units": units},
        ),
    }
    if not subhourly:
        variables["time_bnds"] = (
            ("time", "bnds"),
            np.stack((points - 1800, points + 1800), axis=1),
        )
    path = tmp_path / "data.nc"
    xr.Dataset(variables, coords=coords).to_netcdf(path)
    config = _catalog_config(path, "v1", "200001-201912")
    if subhourly:
        config["facets"].pop("timerange")
    catalog = path.with_suffix(".json")
    metadata = json.loads(catalog.read_text())
    metadata["catalog_dict"][0].update(table_id=mip, variable_id=short_name)
    if not subhourly:
        # CMIP filenames can give point years rather than averaging bounds.
        metadata["catalog_dict"][0]["time_range"] = (
            f"200501-{2005 + 10 * (samples - 1)}01"
        )
    catalog.write_text(json.dumps(metadata))
    session["projects"]["CMIP6"]["data"] = {"catalog": config}
    dataset = Dataset(
        project="CMIP6",
        **{**FACETS, "mip": mip, "short_name": short_name},
        frequency=frequency,
        timerange=timerange,
    )
    dataset.session = session
    check.data_availability(dataset)
    if error:
        with pytest.raises(InputFilesNotFound, match=error):
            dataset.load()
    else:
        assert dataset.load().shape == shape


def test_catalog_relative_and_explicit_time_ranges_agree(
    tmp_path: Path,
    session: Session,
) -> None:
    """A duration must select the same monthly assets as its explicit period."""
    full = tmp_path / "input" / "full.nc"
    _write_asset(full, (2001, 2001), 0.0)
    rows = []
    with xr.open_dataset(full, decode_cf=False) as data:
        for month in range(1, 13):
            asset = tmp_path / f"tas_2001{month:02d}.nc"
            data.isel(time=slice(month - 1, month)).to_netcdf(asset)
            config = _catalog_config(
                asset,
                "v1",
                f"2001{month:02d}-2001{month:02d}",
            )
            metadata = json.loads(asset.with_suffix(".json").read_text())
            rows.extend(metadata["catalog_dict"])
    metadata["catalog_dict"] = rows
    metadata["aggregation_control"]["aggregations"] = [
        {
            "type": "join_existing",
            "attribute_name": "time_range",
            "options": {"dim": "time"},
        },
    ]
    catalog = tmp_path / "monthly.json"
    catalog.write_text(json.dumps(metadata))
    config["catalog"] = str(catalog)
    session["projects"]["CMIP6"]["data"] = {"catalog": config}
    outputs = []
    for timerange in ("200101/200112", "P1Y/2001"):
        dataset = Dataset(
            project="CMIP6",
            **FACETS,
            frequency="mon",
            timerange=timerange,
        )
        dataset.session = session
        dataset.find_files()
        element = dataset.files[0]
        assert isinstance(element, IntakeEsmDataset)
        assert len(element.catalog.df) == 12
        outputs.append(dataset.load())
    np.testing.assert_array_equal(outputs[0].data, outputs[1].data)
    np.testing.assert_array_equal(
        outputs[0].coord("time").points,
        outputs[1].coord("time").points,
    )
