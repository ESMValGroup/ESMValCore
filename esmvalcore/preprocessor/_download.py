"""Functions for downloading climate data files."""
import logging
import os
import subprocess

from .._data_finder import get_start_end_year, select_files

logger = logging.getLogger(__name__)


def _synda_search_cmd(variable):
    """Create a synda command for searching for variable."""
    project = variable.get('project', '')
    if project == 'CMIP5':
        query = {
            'project': 'CMIP5',
            'cmor_table': variable.get('mip'),
            'variable': variable.get('short_name'),
            'model': variable.get('dataset'),
            'experiment': variable.get('exp'),
            'ensemble': variable.get('ensemble'),
        }
    elif project == 'CMIP6':
        query = {
            'project': 'CMIP6',
            'activity_id': variable.get('activity'),
            'table_id': variable.get('mip'),
            'variable_id': variable.get('short_name'),
            'source_id': variable.get('dataset'),
            'experiment_id': variable.get('exp'),
            'variant_label': variable.get('ensemble'),
            'grid_label': variable.get('grid'),
        }
    else:
        raise NotImplementedError(
            f"Unknown project {project}, unable to download data.")

    query = {facet: value for facet, value in query.items() if value}

    query = ("{}='{}'".format(facet, value) for facet, value in query.items())

    cmd = ['synda', 'search', '--file']
    cmd.extend(query)
    cmd = ' '.join(cmd)
    return cmd


def synda_search(variable):
    """Search files using synda."""
    cmd = _synda_search_cmd(variable)
    logger.debug("Running: %s", cmd)
    result = subprocess.check_output(cmd, shell=True, universal_newlines=True)
    logger.debug('Result:\n%s', result.strip())

    files = [
        line.split()[-1] for line in result.split('\n')
        if line.startswith('new')
    ]
    if variable.get('frequency', '') != 'fx':
        files = select_files(files, variable['start_year'],
                             variable['end_year'])

        # filter partially overlapping files
        intervals = {get_start_end_year(name): name for name in files}
        files = []
        for (start, end), filename in intervals.items():
            for _start, _end in intervals:
                if start == _start and end == _end:
                    continue
                if start >= _start and end <= _end:
                    break
            else:
                files.append(filename)

    logger.debug("Selected files:\n%s", '\n'.join(files))

    return files


def synda_download(synda_name, dest_folder):
    """Download file using synda."""
    filename = '.'.join(synda_name.split('.')[-2:])
    local_file = os.path.join(dest_folder, filename)

    if not os.path.exists(local_file):
        cmd = [
            'synda', 'get', '--dest_folder={}'.format(dest_folder),
            '--verify_checksum', synda_name
        ]
        cmd = ' '.join(cmd)
        logger.debug("Running: %s", cmd)
        subprocess.check_call(cmd, shell=True)

    return local_file


def download(files, dest_folder):
    """Download files that are not available locally."""
    os.makedirs(dest_folder, exist_ok=True)

    local_files = []
    for name in files:
        local_file = synda_download(synda_name=name, dest_folder=dest_folder)
        local_files.append(local_file)

    return local_files
