from esmvalcore._data_finder import select_files


def test_select_files():

    files = [
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_195501-195912.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196001-196412.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196501-196912.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_197001-197412.nc",
    ]

    result = select_files(files, 1962, 1967)

    expected = [
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196001-196412.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196501-196912.nc",
    ]

    assert result == expected
