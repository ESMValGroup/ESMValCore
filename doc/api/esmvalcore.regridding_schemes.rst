.. _regriddin_schemes:

Regridding schemes
==================

Iris natively supports data regridding with its :meth:`iris.cube.Cube.regrid`
method and a set of predefined regridding schemes provided in the
:mod:`~iris.analysis` module (further details on this are given `here
<https://scitools-iris.readthedocs.io/en/latest/userguide/interpolation_and_regridding.html>`__).
In this module, further regridding schemes are provided that are compatible
with :meth:`iris.cube.Cube.regrid`.

Example:

.. code:: python

   from esmvalcore.preprocessor.regrid_schemes import ESMPyAreaWeighted
   regridded_cube = cube.regrid(target_grid, ESMPyAreaWeighted())

.. automodule:: esmvalcore.preprocessor.regrid_schemes
