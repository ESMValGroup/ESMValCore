from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import intake_esgf
import iris.cube
import pytest

from esmvalcore.dataset import Dataset

if TYPE_CHECKING:
    from esmvalcore.config import Session


def create_temporary_data(tgt_dir: Path) -> None:
    cwd = Path(__file__).parent
    tas_src = cwd / "tas.nc"
    areacella_src = cwd / "areacella.nc"

    tas_tgt = Path(
        tgt_dir
        / "CMIP6"
        / "CMIP"
        / "CCCma"
        / "CanESM5"
        / "historical"
        / "r1i1p1f1"
        / "Amon"
        / "tas"
        / "gn"
        / "v20190429"
        / "tas_Amon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc",
    )
    areacella_tgt = Path(
        tgt_dir
        / "CMIP6"
        / "CMIP"
        / "CCCma"
        / "CanESM5"
        / "historical"
        / "r1i1p1f1"
        / "fx"
        / "areacella"
        / "gn"
        / "v20190429"
        / "areacella_fx_CanESM5_historical_r1i1p1f1_gn.nc",
    )

    tas_tgt.parent.mkdir(parents=True, exist_ok=True)
    tas_tgt.symlink_to(tas_src)

    areacella_tgt.parent.mkdir(parents=True, exist_ok=True)
    areacella_tgt.symlink_to(areacella_src)


@pytest.fixture
def local_data_source(tmp_path: Path) -> dict[str, str]:
    rootpath = tmp_path / "climate_data"
    create_temporary_data(rootpath)
    return {
        "type": "esmvalcore.io.local.LocalDataSource",
        "rootpath": str(rootpath),
        "dirname_template": "{project}/{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/{grid}/{version}",
        "filename_template": "{short_name}_{mip}_{dataset}_{exp}_{ensemble}_{grid}*.nc",
    }


@pytest.fixture
def intake_esgf_data_source(
    tmp_path: Path,
) -> dict[str, str | dict[str, str] | int]:
    rootpath = tmp_path / "local_esgf_cache"
    create_temporary_data(rootpath)
    facets = {
        "activity": "activity_drs",
        "dataset": "source_id",
        "ensemble": "member_id",
        "exp": "experiment_id",
        "institute": "institution_id",
        "grid": "grid_label",
        "mip": "table_id",
        "project": "project",
        "short_name": "variable_id",
    }
    with intake_esgf.conf.set(esg_dataroot=[rootpath], local_cache=[rootpath]):
        return {
            "type": "esmvalcore.io.intake_esgf.IntakeESGFDataSource",
            "facets": facets,
            "priority": 2,
        }


@pytest.mark.parametrize("timerange", ["1850/185002", "*", "*/P2M", "1860/*"])
def test_find_data_local(
    local_data_source: dict[str, str],
    session: Session,
    timerange: str,
) -> None:
    tas = Dataset(
        short_name="tas",
        mip="Amon",
        project="CMIP6",
        dataset="CanESM5",
        ensemble="r1i1p1f1",
        exp="historical",
        grid="gn",
        timerange=timerange,
    )
    tas.add_supplementary(short_name="areacella", mip="fx")
    tas.session = session
    tas.session["projects"]["CMIP6"]["data"] = {
        "local-data-source": local_data_source,
    }

    assert len(tas.files) == 1
    assert "timerange" in tas.files[0].facets
    assert len(tas.supplementaries[0].files) == 1
    assert "timerange" not in tas.supplementaries[0].files


def test_load_local(
    local_data_source: dict[str, str],
    session: Session,
) -> None:
    tas = Dataset(
        short_name="tas",
        mip="Amon",
        project="CMIP6",
        dataset="CanESM5",
        ensemble="r1i1p1f1",
        exp="historical",
        grid="gn",
        timerange="1850/185002",
    )
    tas.add_supplementary(short_name="areacella", mip="fx")
    tas.session = session
    tas.session["projects"]["CMIP6"]["data"] = {
        "local-data-source": local_data_source,
    }

    tas.augment_facets()

    tas.find_files()
    print(tas.files)

    cube = tas.load()

    assert isinstance(cube, iris.cube.Cube)
    assert cube.cell_measures()


@pytest.mark.online
def test_find_data_intake_esgf(
    intake_esgf_data_source: dict[str, str],
    session: Session,
) -> None:
    tas = Dataset(
        short_name="tas",
        mip="Amon",
        project="CMIP6",
        dataset="CanESM5",
        ensemble="r1i1p1f1",
        grid="gn",
        exp="historical",
    )
    tas.add_supplementary(short_name="areacella", mip="fx")
    tas.session = session
    tas.session["projects"]["CMIP6"]["data"] = {
        "intake-esgf-data-source": intake_esgf_data_source,
    }

    assert len(tas.files) == 1
    assert len(tas.supplementaries[0].files) == 1


@pytest.mark.online
def test_load_intake_esgf(
    intake_esgf_data_source: dict[str, str],
    session: Session,
) -> None:
    tas = Dataset(
        short_name="tas",
        mip="Amon",
        project="CMIP6",
        dataset="CanESM5",
        ensemble="r1i1p1f1",
        exp="historical",
        grid="gn",
        timerange="1850/185002",
    )
    tas.add_supplementary(short_name="areacella", mip="fx")
    tas.session = session
    tas.session["projects"]["CMIP6"]["data"] = {
        "intake-esgf-data-source": intake_esgf_data_source,
    }

    tas.augment_facets()

    tas.find_files()
    print(tas.files)

    cube = tas.load()

    assert isinstance(cube, iris.cube.Cube)
    assert cube.cell_measures()


@pytest.mark.online
@pytest.mark.parametrize(
    ("search_data", "n_files"),
    [("quick", 1), ("complete", 2)],
)
def test_find_data_multiple_data_sources(
    intake_esgf_data_source: dict[str, str],
    local_data_source: dict[str, str],
    session: Session,
    search_data: Literal["quick", "complete"],
    n_files: int,
) -> None:
    session["search_data"] = search_data
    tas = Dataset(
        short_name="tas",
        mip="Amon",
        project="CMIP6",
        dataset="CanESM5",
        ensemble="r1i1p1f1",
        grid="gn",
        exp="historical",
    )
    tas.add_supplementary(short_name="areacella", mip="fx")
    tas.session = session
    tas.session["projects"]["CMIP6"]["data"] = {
        "intake-esgf-data-source": intake_esgf_data_source,
        "local-data-source": local_data_source,
    }

    assert len(tas.files) == n_files
    assert len(tas.supplementaries[0].files) == n_files
    assert "climate_data" in str(tas.files[0])
    assert "climate_data" in str(tas.supplementaries[0].files[0])
