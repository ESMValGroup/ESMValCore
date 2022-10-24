from esmvalcore._data_finder import select_by_time
from esmvalcore.local import LocalFile


def create_filelist(filenames: list[str]) -> list[LocalFile]:
    """Create a list of LocalFiles with a timerange facet."""
    files = []
    for filename in filenames:
        file = LocalFile._from_path(filename, drs='', try_timerange=True)
        files.append(file)
    return files


def test_select_by_time():

    filenames = [
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_195501-195912.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196001-196412.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196501-196912.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_197001-197412.nc",
    ]
    files = create_filelist(filenames)

    result = select_by_time(files, '1962/1967')

    expected = [
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196001-196412.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196501-196912.nc",
    ]

    assert result == [LocalFile(f) for f in expected]


def test_select_by_time_monthly_resolution():
    """Test file selection works for monthly data."""
    filenames = [
        "pr_Amon_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_196011-196110.nc",
        "pr_Amon_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_196111-196210.nc",
        "pr_Amon_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_196211-196310.nc",
        "pr_Amon_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_196311-196410.nc",
    ]
    files = create_filelist(filenames)

    result = select_by_time(files, '196201/196205')

    expected = [
        "pr_Amon_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_196111-196210.nc"
    ]

    assert result == [LocalFile(f) for f in expected]


def test_select_by_time_daily_resolution():
    """Test file selection works for daily data."""
    filename = "tas_day_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_"

    filenames = [
        filename + "19601101-19611031.nc",
        filename + "19611101-19621031.nc",
        filename + "19621101-19631031.nc"
    ]
    files = create_filelist(filenames)

    result = select_by_time(files, '19600101/19611215')

    expected = [
        filename + "19601101-19611031.nc",
        filename + "19611101-19621031.nc",
    ]
    assert result == [LocalFile(f) for f in expected]


def test_select_by_time_sub_daily_resolution():
    """Test file selection works for sub-daily data."""
    filename = "psl_6hrPlev_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_"

    filenames = [
        filename + "19601101T0900-19611031T2100.nc",
        filename + "19611101T0900-19621031T2100.nc",
        filename + "19621101T0300-19631031T2100.nc",
    ]
    files = create_filelist(filenames)

    result = select_by_time(files, '19600101T0900/19610101T0900')

    expected = [
        filename + "19601101T0900-19611031T2100.nc",
    ]

    assert result == [LocalFile(f) for f in expected]


def test_select_by_time_sub_daily_resolution_no_separator():
    """Test file selection works for sub-daily data."""
    filename = "psl_6hrPlev_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_"

    filenames = [
        filename + "196011010900-196110312100.nc",
        filename + "196111010900-196210312100.nc",
        filename + "196211010300-196310312100.nc",
    ]
    files = create_filelist(filenames)

    result = select_by_time(files, '19600101T0900/19610101T09HH00MM')

    expected = [
        filename + "196011010900-196110312100.nc",
    ]

    assert result == [LocalFile(f) for f in expected]


def test_select_by_time_time_period():
    """Test file selection works with time range given as duration periods of
    various resolution."""
    filename = "pr_Amon_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_"

    filenames = [
        filename + "196011-196110.nc",
        filename + "196111-196210.nc",
        filename + "196211-196310.nc",
        filename + "196311-196410.nc",
        filename + "196411-196510.nc",
    ]
    files = create_filelist(filenames)

    result = select_by_time(files, '196211/P2Y5M')

    expected = [
        filename + "196211-196310.nc",
        filename + "196311-196410.nc",
        filename + "196411-196510.nc",
    ]

    assert result == [LocalFile(f) for f in expected]


def test_select_by_time_datetime_period():
    """Test file selection works with time range given as duration periods of
    various resolution."""
    filename = (
        "psl_6hrPlev_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_")

    filenames = [
        filename + "196011010900-196110312100.nc",
        filename + "196111010900-196210312100.nc",
        filename + "196211010300-196310312100.nc",
    ]
    files = create_filelist(filenames)

    result = select_by_time(
        files,
        '19601101T1300/P1Y0M0DT6H',
    )

    expected = [
        filename + "196011010900-196110312100.nc",
        filename + "196111010900-196210312100.nc",
    ]

    assert result == [LocalFile(f) for f in expected]


def test_select_by_time_varying_format():
    """Test file selection works with time range of various time resolutions
    and formats."""
    filename = "psl_6hrPlev_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_"

    filenames = [
        filename + "196011010900-196110312100.nc",
        filename + "196111010900-196210312100.nc",
        filename + "196211010300-196310312100.nc",
    ]
    files = create_filelist(filenames)

    result_yearly = select_by_time(files, '1960/1962')
    result_monthly = select_by_time(files, '196011/196210')
    result_daily = select_by_time(files, '19601101/19601105')

    assert result_yearly == files
    assert result_monthly == files[0:2]
    assert result_daily == [files[0]]
