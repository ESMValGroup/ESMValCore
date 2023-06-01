import pytest
import yaml

from esmvalcore.config import _dask


def test_get_no_distributed_client(mocker, tmp_path):
    mocker.patch.object(_dask, 'CONFIG_FILE', tmp_path / 'nonexistent.yml')
    with _dask.get_distributed_client() as client:
        assert client is None


@pytest.mark.parametrize('warn_unused_args', [False, True])
def test_get_distributed_client_external(mocker, tmp_path, warn_unused_args):
    # Create mock client configuration.
    cfg = {
        'client': {
            'address': 'tcp://127.0.0.1:42021',
        },
    }
    if warn_unused_args:
        cfg['cluster'] = {'n_workers': 2}
    cfg_file = tmp_path / 'dask.yml'
    with cfg_file.open('w', encoding='utf-8') as file:
        yaml.safe_dump(cfg, file)
    mocker.patch.object(_dask, 'CONFIG_FILE', cfg_file)

    # Create mock distributed.Client
    mock_client = mocker.Mock()
    mocker.patch.object(_dask,
                        'Client',
                        create_autospec=True,
                        return_value=mock_client)

    with _dask.get_distributed_client() as client:
        assert client is mock_client
    _dask.Client.assert_called_with(**cfg['client'])
    mock_client.close.assert_called()


def test_get_distributed_client_slurm(mocker, tmp_path):
    cfg = {
        'cluster': {
            'type': 'dask_jobqueue.SLURMCluster',
            'queue': 'interactive',
            'cores': '8',
            'memory': '16GiB',
        },
    }
    cfg_file = tmp_path / 'dask.yml'
    with cfg_file.open('w', encoding='utf-8') as file:
        yaml.safe_dump(cfg, file)
    mocker.patch.object(_dask, 'CONFIG_FILE', cfg_file)

    # Create mock distributed.Client
    mock_client = mocker.Mock()
    mocker.patch.object(_dask,
                        'Client',
                        create_autospec=True,
                        return_value=mock_client)

    mock_module = mocker.Mock()
    mock_cluster_cls = mocker.Mock()
    mock_module.SLURMCluster = mock_cluster_cls
    mocker.patch.object(_dask.importlib,
                        'import_module',
                        create_autospec=True,
                        return_value=mock_module)
    with _dask.get_distributed_client() as client:
        assert client is mock_client
    mock_client.close.assert_called()
    mock_cluster = mock_cluster_cls.return_value
    _dask.Client.assert_called_with(address=mock_cluster.scheduler_address)
    args = {k: v for k, v in cfg['cluster'].items() if k != 'type'}
    mock_cluster_cls.assert_called_with(**args)
    mock_cluster.close.assert_called()
