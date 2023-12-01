Find and download files from ESGF
=================================

This module provides the function :py:func:`esmvalcore.esgf.find_files`
for searching for files on ESGF using the ESMValTool vocabulary.
It returns :py:class:`esmvalcore.esgf.ESGFFile` objects, which have a convenient
:py:meth:`esmvalcore.esgf.ESGFFile.download` method for downloading the files.

See :ref:`config-esgf` for instructions on configuring this module.

esmvalcore.esgf
---------------
.. autofunction:: esmvalcore.esgf.find_files
.. autofunction:: esmvalcore.esgf.download
.. autoclass:: esmvalcore.esgf.ESGFFile

esmvalcore.esgf.facets
----------------------
.. automodule:: esmvalcore.esgf.facets
