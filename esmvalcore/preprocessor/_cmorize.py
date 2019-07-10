"""CMORizer for certain projects."""
import importlib.util
import logging
import os

logger = logging.getLogger(__name__)


def cmorize(in_files, short_name, var_mapping, output_dir, cmorizer):
    """Use project-specific CMORizer and CMORizer data."""
    cmorizer = os.path.expanduser(cmorizer)
    if not os.path.isabs(cmorizer):
        root = os.path.dirname(os.path.realpath(__file__))
        cmorizer = os.path.join(root, cmorizer)
    try:
        spec = importlib.util.spec_from_file_location('cmorizer', cmorizer)
        cmorizer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cmorizer_module)
    except Exception:
        logger.error(
            "CMORizer %s given in 'config-developer.yml' is not a valid "
            "CMORizer, make sure that it exists and that it contains a "
            "function called 'cmorize'", cmorizer)
        raise
    if not hasattr(cmorizer_module, 'cmorize'):
        raise ValueError(
            f"CMORizer {cmorizer} does not contain a function called "
            f"'cmorize'")
    logger.debug("Successfully loaded CMORizer %s", cmorizer)
    return cmorizer_module.cmorize(in_files, short_name, var_mapping,
                                   output_dir)
