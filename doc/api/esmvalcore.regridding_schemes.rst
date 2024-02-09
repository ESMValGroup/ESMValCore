.. _regridding_schemes:

Regridding schemes
==================

Iris natively supports data regridding with its :meth:`iris.cube.Cube.regrid`
method and a set of predefined regridding schemes provided in the
:mod:`~iris.analysis` module (further details are given on `this
<https://scitools-iris.readthedocs.io/en/latest/userguide/interpolation_and_regridding.html>`__
page).
Here, further regridding schemes are provided that are compatible with
:meth:`iris.cube.Cube.regrid`.

Example:

.. code:: python

   from esmvalcore.preprocessor.regrid_schemes import ESMPyAreaWeighted

   regridded_cube = cube.regrid(target_grid, ESMPyAreaWeighted())

.. automodule:: esmvalcore.preprocessor.regrid_schemes
   :no-show-inheritance:
