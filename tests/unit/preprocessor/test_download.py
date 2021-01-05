import pytest

import esmvalcore.preprocessor
from esmvalcore.preprocessor import _download


@pytest.mark.parametrize(
    'variable, cmd',
    [
        (
            {
                'dataset': 'CanESM2',
                'ensemble': 'r1i1p1',
                'exp': 'historical',
                'mip': 'Amon',
                'project': 'CMIP5',
                'variable': 'ta',
            },
            ("synda search --file"
             " project='CMIP5'"
             " cmor_table='Amon'"
             " model='CanESM2'"
             " experiment='historical'"
             " ensemble='r1i1p1'"),
        ),
        (
            {
                'activity': 'CMIP',
                'dataset': 'BCC-ESM1',
                'ensemble': 'r1i1p1f1',
                'exp': 'historical',
                'grid': 'gn',
                'mip': 'Amon',
                'project': 'CMIP6',
                'variable': 'ta',
            },
            ("synda search --file"
             " project='CMIP6'"
             " activity_id='CMIP'"
             " table_id='Amon'"
             " source_id='BCC-ESM1'"
             " experiment_id='historical'"
             " variant_label='r1i1p1f1'"
             " grid_label='gn'"),
        ),
    ],
)
def test_synda_search_cmd(variable, cmd):
    assert _download._synda_search_cmd(variable) == cmd


def test_synda_search_cmd_fail_unknown_project():

    with pytest.raises(NotImplementedError):
        _download._synda_search_cmd({'project': 'Unknown'})


def test_synda_search(mocker):

    variable = {
        'frequency': 'mon',
        'start_year': 1962,
        'end_year': 1966,
    }

    cmd = mocker.sentinel.cmd

    dataset = ("CMIP6.CMIP.MPI-M.MPI-ESM1-2-HR.historical"
               ".r1i1p1f1.Amon.pr.gn.v20190710")
    files = [
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_195501-195912.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196001-196412.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196501-197212.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196501-196912.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_197001-197412.nc",
    ]

    all_files = [f"{dataset}.{filename}" for filename in files]
    selected_files = all_files[1:4]

    mocker.patch.object(_download,
                        '_synda_search_cmd',
                        return_value=cmd,
                        autospec=True)
    mocker.patch.object(_download.subprocess,
                        'check_output',
                        return_value="\n".join(
                            f"new  12.7 MB  {dataset}.{filename}"
                            for filename in files),
                        autospec=True)
    mocker.patch.object(_download,
                        'select_files',
                        return_value=selected_files,
                        autospec=True)
    mocker.patch.object(_download,
                        'get_start_end_year',
                        side_effect=[(1960, 1964), (1965, 1972), (1965, 1969)],
                        autospec=True)

    result = _download.synda_search(variable)

    # Check calls and result
    _download._synda_search_cmd.assert_called_once_with(variable)
    _download.subprocess.check_output.assert_called_once_with(
        cmd, shell=True, universal_newlines=True)
    _download.select_files.assert_called_once_with(all_files,
                                                   variable['start_year'],
                                                   variable['end_year'])
    _download.get_start_end_year.assert_has_calls(
        [mocker.call(filename) for filename in selected_files])
    assert result == all_files[1:3]


@pytest.mark.parametrize('download', [True, False])
def test_synda_download(download, mocker, tmp_path):
    filename = "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_195501-195912.nc"
    local_path = tmp_path / filename
    synda_path = ("CMIP6.CMIP.MPI-M.MPI-ESM1-2-HR.historical"
                  f".r1i1p1f1.Amon.pr.gn.v20190710.{filename}")
    cmd = f'synda get --dest_folder={tmp_path} --verify_checksum {synda_path}'

    mocker.patch.object(_download.subprocess, 'check_call', autospec=True)

    if not download:
        local_path.touch()

    result = _download.synda_download(synda_path, tmp_path)
    if download:
        _download.subprocess.check_call.assert_called_once_with(cmd,
                                                                shell=True)
    else:
        _download.subprocess.check_call.assert_not_called()
    assert result == str(local_path)


def test_download(mocker, tmp_path):

    dataset = ("CMIP6.CMIP.MPI-M.MPI-ESM1-2-HR.historical"
               ".r1i1p1f1.Amon.pr.gn.v20190710")
    files = [
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_195501-195912.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196001-196412.nc",
        "pr_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_196501-196912.nc",
    ]

    synda_files = [f"{dataset}.{filename}" for filename in files]
    local_files = [str(tmp_path / filename) for filename in files]

    mocker.patch.object(_download,
                        'synda_download',
                        autospec=True,
                        side_effect=local_files)

    result = esmvalcore.preprocessor.download(synda_files, tmp_path)
    _download.synda_download.assert_has_calls(
        [mocker.call(filename, tmp_path) for filename in synda_files])
    assert result == local_files
