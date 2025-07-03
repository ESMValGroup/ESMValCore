Find and download files from ESGF
=================================

This module provides the function :py:func:`esmvalcore.esgf.find_files`
for searching for files on ESGF using the ESMValTool vocabulary.
It returns :py:class:`esmvalcore.esgf.ESGFFile` objects, which have a convenient
:py:meth:`esmvalcore.esgf.ESGFFile.download` method for downloading the file
as well as a :func:`esmvalcore.esgf.download` function for downloading multiple
files in parallel.

See :ref:`config-esgf` for instructions on configuring this module.

esmvalcore.esgf
---------------
.. automodule:: esmvalcore.esgf
    :noindex:

esmvalcore.esgf.facets
----------------------
.. automodule:: esmvalcore.esgf.facets
