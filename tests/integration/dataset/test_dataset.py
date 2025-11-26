from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import iris.cube
import pytest

from esmvalcore.dataset import Dataset

if TYPE_CHECKING:
    from esmvalcore.config import Session


@pytest.fixture
def example_data_source(tmp_path: Path) -> dict[str, str]:
    cwd = Path(__file__).parent
    tas_src = cwd / "tas.nc"
    areacella_src = cwd / "areacella.nc"

    rootpath = tmp_path / "climate_data"
    tas_tgt = (
        rootpath
        / "cmip5"
        / "output1"
        / "CCCma"
        / "CanESM2"
        / "historical"
        / "mon"
        / "atmos"
        / "Amon"
        / "r1i1p1"
        / "v20120718"
        / "tas_Amon_CanESM2_historical_r1i1p1_185001-200512.nc"
    )
    areacella_tgt = (
        rootpath
        / "cmip5"
        / "output1"
        / "CCCma"
        / "CanESM2"
        / "historical"
        / "fx"
        / "atmos"
        / "fx"
        / "r0i0p0"
        / "v20120410"
        / "areacella_fx_CanESM2_historical_r0i0p0.nc"
    )

    tas_tgt.parent.mkdir(parents=True, exist_ok=True)
    tas_tgt.symlink_to(tas_src)

    areacella_tgt.parent.mkdir(parents=True, exist_ok=True)
    areacella_tgt.symlink_to(areacella_src)
    return {
        "type": "esmvalcore.local.LocalDataSource",
        "rootpath": str(rootpath),
        "dirname_template": "{project.lower}/{product}/{institute}/{dataset}/{exp}/{frequency}/{modeling_realm}/{mip}/{ensemble}/{version}",
        "filename_template": "{short_name}_{mip}_{dataset}_{exp}_{ensemble}*.nc",
    }


@pytest.mark.parametrize("timerange", ["1850/185002", "*", "*/P2M", "1860/*"])
def test_find_data(
    example_data_source: dict[str, str],
    session: Session,
    timerange: str,
) -> None:
    tas = Dataset(
        short_name="tas",
        mip="Amon",
        project="CMIP5",
        dataset="CanESM2",
        ensemble="r1i1p1",
        exp="historical",
        timerange=timerange,
    )
    tas.add_supplementary(short_name="areacella", mip="fx", ensemble="r0i0p0")
    tas.session = session
    tas.session["projects"]["CMIP5"]["data"] = {
        "example-data-source": example_data_source,
    }

    assert len(tas.files) == 1
    assert "timerange" in tas.files[0].facets
    assert len(tas.supplementaries[0].files) == 1
    assert "timerange" not in tas.supplementaries[0].files


def test_load(example_data_source: dict[str, str], session: Session) -> None:
    tas = Dataset(
        short_name="tas",
        mip="Amon",
        project="CMIP5",
        dataset="CanESM2",
        ensemble="r1i1p1",
        exp="historical",
        timerange="1850/185002",
    )
    tas.add_supplementary(short_name="areacella", mip="fx", ensemble="r0i0p0")
    tas.session = session
    tas.session["projects"]["CMIP5"]["data"] = {
        "example-data-source": example_data_source,
    }

    tas.augment_facets()

    tas.find_files()
    print(tas.files)

    cube = tas.load()

    assert isinstance(cube, iris.cube.Cube)
    assert cube.cell_measures()
