"""Find files on the ESGF and download them.

.. note::

    This module uses `esgf-pyclient <https://esgf-pyclient.readthedocs.io>`_
    to search for and download files from the Earth System Grid Federation (ESGF).
    `esgf-pyclient`_ uses a
    `deprecated API <https://esgf.github.io/esg-search/ESGF_Search_RESTful_API.html>`__
    that is scheduled to be taken offline and replaced by new APIs based on
    STAC (ESGF East) and Globus (ESGF West). An ESGF node mimicking the deprecated
    API but built op top of Globus will be kept online for some time at
    https://esgf-node.ornl.gov/esgf-1-5-bridge, but users are encouraged
    to migrate to the new APIs as soon as possible by using the
    :mod:`esmvalcore.io.intake_esgf` module instead.

This module provides the function :py:func:`esmvalcore.esgf.find_files`
for searching for files on ESGF using the ESMValTool vocabulary.
It returns :class:`esmvalcore.esgf.ESGFFile` objects, which have a convenient
:meth:`esmvalcore.esgf.ESGFFile.download` method for downloading the file.
A :func:`esmvalcore.esgf.download` function for downloading multiple files in
parallel is also available.

It also provides an :class:`esmvalcore.esgf.ESGFDataSource` that can be
used to find files on ESGF from the :class:`~esmvalcore.dataset.Dataset`
or the :ref:`recipe <recipe>`. To use it, run the command

.. code:: bash

    esmvalcore config copy data-esmvalcore-esgf.yml

to copy the default configuration file for this module to your configuration
directory. This will create a file with the following content:

.. literalinclude:: ../configurations/data-esmvalcore-esgf.yml
   :language: yaml

See :ref:`config-esgf` for instructions on additional configuration
options of this module.
"""

from esmvalcore.esgf._download import ESGFFile, download
from esmvalcore.esgf._search import ESGFDataSource, find_files

__all__ = [
    "ESGFFile",
    "ESGFDataSource",
    "download",
    "find_files",
]
