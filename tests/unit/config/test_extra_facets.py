import textwrap
from importlib.resources import files as importlib_files

import pytest

from esmvalcore.config import CFG
from esmvalcore.config._extra_facets import (
    _deep_update,
    _load_extra_facets,
    get_extra_facets,
)
from esmvalcore.dataset import Dataset

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


# TODO: Remove in v2.15.0
@pytest.mark.parametrize(
    ("project", "extra_facets_dir", "expected"),
    TEST_LOAD_EXTRA_FACETS,
)
def test_load_extra_facets(project, extra_facets_dir, expected):
    extra_facets = _load_extra_facets(project, extra_facets_dir)
    assert extra_facets == expected


# TODO: Remove in v2.15.0
def test_get_deprecated_extra_facets(tmp_path, monkeypatch):
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
    monkeypatch.setitem(CFG, "extra_facets_dir", [str(tmp_path)])

    extra_facets = get_extra_facets(dataset)

    assert extra_facets == {"key": "value"}


def test_get_extra_facets_cmip3():
    dataset = Dataset(
        project="CMIP3",
        mip="A1",
        short_name="tas",
        dataset="CM3",
    )
    extra_facets = get_extra_facets(dataset)

    assert extra_facets == {"institute": ["CNRM", "INM", "CNRM_CERFACS"]}


def test_get_extra_facets_cmip5():
    dataset = Dataset(
        project="CMIP5",
        mip="Amon",
        short_name="tas",
        dataset="ACCESS1-0",
    )
    extra_facets = get_extra_facets(dataset)

    assert extra_facets == {
        "institute": ["CSIRO-BOM"],
        "product": ["output1", "output2"],
    }
