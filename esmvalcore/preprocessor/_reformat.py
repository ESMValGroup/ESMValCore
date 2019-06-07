"""Simple interface to reformat and CMORize functions."""
from ..cmor.check import cmor_check, cmor_check_data, cmor_check_metadata
from ..cmor.fix import fix_data, fix_file, fix_metadata


__all__ = [
    'cmor_check',
    'cmor_check_metadata',
    'cmor_check_data',
    'fix_file',
    'fix_metadata',
    'fix_data',
]
