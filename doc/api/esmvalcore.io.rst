Access data from any source
===========================

ESMValCore supports a modular system for reading data from various data sources.
In the future, this module may be extended with support for writing output data.

The interface is defined in the :mod:`esmvalcore.io.protocol` module and
the other modules here provide an implementation for a particular data source.

.. toctree::
   :maxdepth: 1

   esmvalcore.io.protocol
   esmvalcore.io.intake_esgf

esmvalcore.io
-------------
.. automodule:: esmvalcore.io
