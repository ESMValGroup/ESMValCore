from esmvalcore._data_finder import select_files


def test_select_files():

    files = [
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_195501-195912.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196001-196412.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196501-196912.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_197001-197412.nc",
    ]

    result = select_files(files, '1962/1967')

    expected = [
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196001-196412.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196501-196912.nc",
    ]

    assert result == expected


def test_select_files_monthly_resolution():
    """Test file selection works for monthly data."""
    files = [
        "pr_Amon_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_196011-196110.nc",
        "pr_Amon_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_196111-196210.nc",
        "pr_Amon_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_196211-196310.nc",
        "pr_Amon_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_196311-196410.nc",
    ]

    result = select_files(files, '196201/196205')

    expected = [
        "pr_Amon_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_196111-196210.nc"
    ]

    assert result == expected


def test_select_files_daily_resolution():
    """Test file selection works for daily data."""
    filename = "tas_day_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_"

    files = [
        filename + "19601101-19611031.nc",
        filename + "19611101-19621031.nc",
        filename + "19621101-19631031.nc"
    ]

    result = select_files(files, '19600101/19611215')

    expected = [
        filename + "19601101-19611031.nc",
        filename + "19611101-19621031.nc",
    ]
    assert result == expected


def test_select_files_sub_daily_resolution():
    """Test file selection works for sub-daily data."""
    filename = "psl_6hrPlev_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_"

    files_no_separator = [
        filename + "196011010900-196110312100.nc",
        filename + "196111010900-196210312100.nc",
        filename + "196211010300-196310312100.nc",
    ]

    files_separator = [
        filename + "19601101T0900-19611031T2100.nc",
        filename + "19611101T0900-19621031T2100.nc",
        filename + "19621101T0300-19631031T2100.nc",
    ]

    result_no_separator = select_files(
        files_no_separator,
        '19600101T0900/19610101T09HH00MM')
    result_separator = select_files(
        files_separator,
        '19600101T0900/19610101T0900')

    expected_no_separator = [
        filename + "196011010900-196110312100.nc",
    ]

    expected_separator = [
        filename + "19601101T0900-19611031T2100.nc",
    ]

    assert result_no_separator == expected_no_separator
    assert result_separator == expected_separator


def test_select_files_time_period():
    """Test file selection works with time range given as duration periods of
    various resolution."""
    filename_date = "pr_Amon_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_"
    filename_datetime = (
        "psl_6hrPlev_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_")

    files_date = [
        filename_date + "196011-196110.nc",
        filename_date + "196111-196210.nc",
        filename_date + "196211-196310.nc",
        filename_date + "196311-196410.nc",
        filename_date + "196411-196510.nc",
    ]

    files_datetime = [
        filename_datetime + "196011010900-196110312100.nc",
        filename_datetime + "196111010900-196210312100.nc",
        filename_datetime + "196211010300-196310312100.nc",
    ]

    result_date = select_files(files_date, '196211/P2Y5M')
    result_datetime = select_files(files_datetime, '19601101T1300/P1Y0M0DT6H')

    expected_date = [
        filename_date + "196211-196310.nc",
        filename_date + "196311-196410.nc",
        filename_date + "196411-196510.nc",
    ]

    expected_datetime = [
        filename_datetime + "196011010900-196110312100.nc",
        filename_datetime + "196111010900-196210312100.nc",
    ]

    assert result_date == expected_date
    assert result_datetime == expected_datetime


def test_select_files_varying_format():
    """Test file selection works with time range of various time resolutions
    and formats."""
    filename = "psl_6hrPlev_EC-Earth3_dcppA-hindcast_s1960-r1i1p1f1_gr_"

    files = [
        filename + "196011010900-196110312100.nc",
        filename + "196111010900-196210312100.nc",
        filename + "196211010300-196310312100.nc",
    ]

    result_yearly = select_files(files, '1960/1962')
    result_monthly = select_files(files, '196011/196210')
    result_daily = select_files(files, '19601101/19601105')

    assert result_yearly == files
    assert result_monthly == files[0:2]
    assert result_daily == [files[0]]
