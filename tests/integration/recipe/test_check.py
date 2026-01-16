"""Integration tests for :mod:`esmvalcore._recipe.check`."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pyesgf.search.results
import pytest

import esmvalcore._recipe.check
import esmvalcore.io.esgf
from esmvalcore._recipe import check
from esmvalcore.dataset import Dataset
from esmvalcore.exceptions import RecipeError
from esmvalcore.io.local import LocalFile
from esmvalcore.preprocessor import PreprocessorFile

if TYPE_CHECKING:
    import pytest_mock


def test_ncl_version(mocker):
    ncl = "/path/to/ncl"
    mocker.patch.object(
        check,
        "which",
        autospec=True,
        return_value=ncl,
    )
    mocker.patch.object(
        check.subprocess,
        "check_output",
        autospec=True,
        return_value="6.6.2\n",
    )
    check.ncl_version()


def test_ncl_version_too_low(mocker):
    ncl = "/path/to/ncl"
    mocker.patch.object(
        check,
        "which",
        autospec=True,
        return_value=ncl,
    )
    mocker.patch.object(
        check.subprocess,
        "check_output",
        autospec=True,
        return_value="6.3.2\n",
    )
    with pytest.raises(
        RecipeError,
        match=r"NCL version 6.4 or higher is required",
    ):
        check.ncl_version()


def test_ncl_version_no_ncl(mocker):
    mocker.patch.object(
        check,
        "which",
        autospec=True,
        return_value=None,
    )
    with pytest.raises(
        RecipeError,
        match="cannot find an NCL installation",
    ):
        check.ncl_version()


def test_ncl_version_broken(mocker):
    ncl = "/path/to/ncl"
    mocker.patch.object(
        check,
        "which",
        autospec=True,
        return_value=ncl,
    )
    mocker.patch.object(
        check.subprocess,
        "check_output",
        autospec=True,
        side_effect=subprocess.CalledProcessError(1, [ncl, "-V"]),
    )
    with pytest.raises(
        RecipeError,
        match="NCL installation appears to be broken",
    ):
        check.ncl_version()


ERR_ALL = "Looked for files matching%s"
ERR_RANGE = "No input data available for years {} in files:\n{}"
VAR = {
    "frequency": "mon",
    "short_name": "tas",
    "timerange": "2020/2025",
    "alias": "alias",
    "start_year": 2020,
    "end_year": 2025,
}
FX_VAR = {
    "frequency": "fx",
    "short_name": "areacella",
}
FILES = [
    "a/b/c_20200101-20201231",
    "a/b/c_20210101-20211231",
    "a/b/c_20220101-20221231",
    "a/b/c_20230101-20231231",
    "a/b/c_20240101-20241231",
    "a/b/c_20250101-20251231",
]

DATA_AVAILABILITY_DATA = [
    (FILES, dict(VAR), None),
    (FILES, dict(FX_VAR), None),
    (FILES[1:], dict(VAR), ERR_RANGE.format("2020", "\n".join(FILES[1:]))),
    (FILES[:-1], dict(VAR), ERR_RANGE.format("2025", "\n".join(FILES[:-1]))),
    (
        FILES[:-3],
        dict(VAR),
        ERR_RANGE.format("2023-2025", "\n".join(FILES[:-3])),
    ),
    (
        [FILES[1], FILES[3]],
        dict(VAR),
        ERR_RANGE.format(
            "2020, 2022, 2024-2025",
            "\n".join([FILES[1], FILES[3]]),
        ),
    ),
]


@pytest.mark.parametrize(
    ("input_files", "var", "error"),
    DATA_AVAILABILITY_DATA,
)
@mock.patch("esmvalcore._recipe.check.logger", autospec=True)
def test_data_availability_data(mock_logger, input_files, var, error):
    """Test check for data when data is present."""
    dataset = Dataset(**var)
    files = []
    for filename in input_files:
        file = LocalFile(filename)
        file.facets["timerange"] = filename.split("_")[-1].replace("-", "/")
        files.append(file)
    dataset.files = files
    if error is None:
        check.data_availability(dataset)
        mock_logger.error.assert_not_called()
    else:
        with pytest.raises(RecipeError) as rec_err:
            check.data_availability(dataset)
        assert str(rec_err.value) == error
    assert dataset.facets == var


def test_data_availability_no_data(
    caplog: pytest.LogCaptureFixture,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test check for data when no data is present."""
    dataset = Dataset(
        frequency="mon",
        short_name="tas",
        timerange="2020/2025",
        alias="alias",
        start_year=2020,
        end_year=2025,
    )
    dataset.files = []
    mock_data_source = mocker.Mock()
    mock_data_source.debug_info = "debug info"
    dataset._used_data_sources = [mock_data_source]
    with pytest.raises(RecipeError) as exc:
        check.data_availability(dataset)
    assert str(exc.value) == "Missing data for Dataset: tas, mon"
    assert len(caplog.records) == 2
    assert caplog.records[0].message == "\n".join(
        [
            f"No files were found for {dataset},",
            "using data sources:",
            f"- data source: {mock_data_source}",
            "  message: debug info",
        ],
    )
    assert (
        caplog.records[1].message
        == "Set 'log_level' to 'debug' to get more information"
    )


GOOD_TIMERANGES = [
    "*",
    "1990/1992",
    "19900101/19920101",
    "19900101T120000/19920101T120000",
    "1990/*",
    "*/1992",
    "1990/P2Y",
    "19900101/P2Y2M1D",
    "19900101T0000/P2Y2M1DT12H00M00S",
    "P2Y/1992",
    "P2Y2M1D/19920101",
    "P2Y2M1D/19920101T120000",
    "P2Y/*",
    "P2Y2M1D/*",
    "P2Y21DT12H00M00S/*",
    "*/P2Y",
    "*/P2Y2M1D",
    "*/P2Y21DT12H00M00S",
    "1/301",
    "1/*",
    "*/301",
]


@pytest.mark.parametrize("timerange", GOOD_TIMERANGES)
def test_valid_time_selection(timerange):
    """Check that good definitions do not raise anything."""
    check.valid_time_selection(timerange)


BAD_TIMERANGES = [
    (
        "randomnonsense",
        "Invalid value encountered for `timerange`. Valid values must be "
        "separated by `/`. Got ['randomnonsense'] instead.",
    ),
    (
        "199035345/19923463164526",
        "Invalid value encountered for `timerange`. Valid value must follow "
        "ISO 8601 standard for dates and duration periods, or be set to '*' "
        "to load available years. Got ['199035345', '19923463164526'] instead.\n"
        "Unrecognised ISO 8601 date format: '199035345'",
    ),
    (
        "P11Y/P42Y",
        "Invalid value encountered for `timerange`. Cannot set both "
        "the beginning and the end as duration periods.",
    ),
    (
        "P11X/19923463164526",
        "Invalid value encountered for `timerange`. "
        "P11X is not valid duration according to ISO 8601.\n"
        "ISO 8601 time designator 'T' missing. "
        "Unable to parse datetime string '11X'",
    ),
    (
        "19923463164526/P11X",
        "Invalid value encountered for `timerange`. "
        "P11X is not valid duration according to ISO 8601.\n"
        "ISO 8601 time designator 'T' missing. "
        "Unable to parse datetime string '11X'",
    ),
]


@pytest.mark.parametrize(("timerange", "message"), BAD_TIMERANGES)
def test_valid_time_selection_rejections(timerange, message):
    """Check that bad definitions raise RecipeError."""
    with pytest.raises(check.RecipeError) as rec_err:
        check.valid_time_selection(timerange)
    assert str(rec_err.value) == message


def test_data_availability_nonexistent(tmp_path):
    var = {
        "dataset": "ABC",
        "short_name": "tas",
        "frequency": "mon",
        "timerange": "1990/1992",
        "start_year": 1990,
        "end_year": 1992,
    }
    result = pyesgf.search.results.FileResult(
        json={
            "dataset_id": "CMIP6.ABC.v1|something.org",
            "dataset_id_template_": ["%(mip_era)s.%(source_id)s"],
            "project": ["CMIP6"],
            "size": 10,
            "title": "tas_1990-1992.nc",
        },
        context=None,
    )
    dest_folder = tmp_path
    input_files = [
        esmvalcore.io.esgf.ESGFFile([result]).local_file(dest_folder),
    ]
    dataset = Dataset(**var)
    dataset.files = input_files
    check.data_availability(dataset)


def test_reference_for_bias_preproc_empty():
    """Test ``reference_for_bias_preproc``."""
    products = {
        PreprocessorFile(filename=Path("10")),
        PreprocessorFile(filename=Path("20")),
        PreprocessorFile(filename=Path("30")),
    }
    check.reference_for_bias_preproc(products)


def test_reference_for_bias_preproc_one_ref():
    """Test ``reference_for_bias_preproc`` with one reference."""
    products = {
        PreprocessorFile(filename=Path("90")),
        PreprocessorFile(filename=Path("10"), settings={"bias": {}}),
        PreprocessorFile(filename=Path("20"), settings={"bias": {}}),
        PreprocessorFile(
            filename=Path("30"),
            settings={"bias": {}},
            attributes={"reference_for_bias": True},
        ),
    }
    check.reference_for_bias_preproc(products)


def test_reference_for_bias_preproc_no_ref():
    """Test ``reference_for_bias_preproc`` with no reference."""
    products = {
        PreprocessorFile(filename=Path("90")),
        PreprocessorFile(filename=Path("10"), settings={"bias": {}}),
        PreprocessorFile(filename=Path("20"), settings={"bias": {}}),
        PreprocessorFile(filename=Path("30"), settings={"bias": {}}),
    }
    with pytest.raises(RecipeError) as rec_err:
        check.reference_for_bias_preproc(products)

    # Note: checking the message directly does not work due to the unknown
    # (machine-dependent) ordering of products in the set
    assert (
        "Expected exactly 1 dataset with 'reference_for_bias: true' in "
        "products\n["
    ) in str(rec_err.value)
    assert "10" in str(rec_err.value)
    assert "20" in str(rec_err.value)
    assert "30" in str(rec_err.value)
    assert "90" not in str(rec_err.value)
    assert (
        "],\nfound 0. Please also ensure that the reference dataset is "
        "not excluded with the 'exclude' option"
    ) in str(rec_err.value)


def test_reference_for_bias_preproc_two_refs():
    """Test ``reference_for_bias_preproc`` with two references."""
    products = {
        PreprocessorFile(filename=Path("90")),
        PreprocessorFile(filename=Path("10"), settings={"bias": {}}),
        PreprocessorFile(
            filename=Path("20"),
            attributes={"reference_for_bias": True},
            settings={"bias": {}},
        ),
        PreprocessorFile(
            filename=Path("30"),
            attributes={"reference_for_bias": True},
            settings={"bias": {}},
        ),
    }
    with pytest.raises(RecipeError) as rec_err:
        check.reference_for_bias_preproc(products)

    # Note: checking the message directly does not work due to the unknown
    # (machine-dependent) ordering of products in the set
    assert (
        "Expected exactly 1 dataset with 'reference_for_bias: true' in "
        "products\n["
    ) in str(rec_err.value)
    assert "10" in str(rec_err.value)
    assert "20" in str(rec_err.value)
    assert "30" in str(rec_err.value)
    assert "90" not in str(rec_err.value)
    assert "],\nfound 2:\n[" in str(rec_err.value)
    assert (
        "].\nPlease also ensure that the reference dataset is "
        "not excluded with the 'exclude' option"
    ) in str(rec_err.value)


INVALID_MM_SETTINGS = {
    "wrong_parametre": "wrong",
    "statistics": ["wrong"],
    "span": "wrong",
    "groupby": "wrong",
    "keep_input_datasets": "wrong",
    "ignore_scalar_coords": "wrong",
}


def test_invalid_multi_model_span():
    with pytest.raises(RecipeError) as rec_err:
        check._verify_span_value(INVALID_MM_SETTINGS["span"])
    assert str(rec_err.value) == (
        "Invalid value encountered for `span` in preprocessor "
        "`multi_model_statistics`. Valid values are ('overlap', 'full')."
        "Got wrong."
    )


def test_invalid_multi_model_groupy():
    with pytest.raises(RecipeError) as rec_err:
        check._verify_groupby(INVALID_MM_SETTINGS["groupby"])
    assert str(rec_err.value) == (
        "Invalid value encountered for `groupby` in preprocessor "
        "`multi_model_statistics`.`groupby` must be defined "
        "as a list. Got wrong."
    )


def test_invalid_multi_model_keep_input():
    with pytest.raises(RecipeError) as rec_err:
        check._verify_keep_input_datasets(
            INVALID_MM_SETTINGS["keep_input_datasets"],
        )
    assert str(rec_err.value) == (
        "Invalid value encountered for `keep_input_datasets`."
        "Must be defined as a boolean (true or false). Got wrong."
    )


def test_invalid_multi_model_ignore_scalar_coords():
    with pytest.raises(RecipeError) as rec_err:
        check._verify_ignore_scalar_coords(
            INVALID_MM_SETTINGS["ignore_scalar_coords"],
        )
    assert str(rec_err.value) == (
        "Invalid value encountered for `ignore_scalar_coords`."
        "Must be defined as a boolean (true or false). Got wrong."
    )
