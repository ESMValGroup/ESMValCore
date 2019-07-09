"""CMORize EMAC data."""
import logging

logger = logging.getLogger(__name__)


def cmorize(in_files, short_name, var_mapping, output_dir):
    """CMORize EMAC data.

    Note
    ----
    At the moment, this is only a "light" CMORizer, i.e. the option
    'light_cmorizer' has to be set in `config-developer.yml` or the CMOR checks
    of the preprocessor will fail.

    """
    return output_dir
