.. _preprocessor_function:

Preprocessor function
*********************

Preprocessor functions are located in :py:mod:`esmvalcore.preprocessor`.
To add a new preprocessor function, start by finding a likely looking file to
add your function to in
`esmvalcore/preprocessor <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/preprocessor>`_.
Create a new file in that directory if you cannot find a suitable place.

The function should look like this:


.. code-block:: python

    def example_preprocessor_function(
        cube,
        example_argument,
        example_optional_argument=5,
    ):
        """Compute an example quantity.

        A more extensive explanation of the computation can be added here. Add
        references to scientific literature if available.

        Parameters
        ----------
        cube: iris.cube.Cube
           Input cube.

        example_argument: str
           Example argument, the value of this argument can be provided in the
           recipe. Describe what valid values are here. In this case, a valid
           argument is the name of a dimension of the input cube.

        example_optional_argument: int, optional
           Another example argument, the value of this argument can optionally
           be provided in the recipe. Describe what valid values are here.

        Returns
        -------
        iris.cube.Cube
          The result of the example computation.
        """

        # Replace this with your own computation
        cube = cube.collapsed(example_argument, iris.analysis.MEAN)

        return cube


The above function needs to be imported in the file
`esmvalcore/preprocessor/__init__.py <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/preprocessor/__init__.py>`__:

.. code-block:: python

    from ._example_module import example_preprocessor_function

    __all__ = [
    ...
    'example_preprocessor_function',
    ...
    ]

The location in the ``__all__`` list above determines the default order in which
preprocessor functions are applied, so carefully consider where you put it
and ask for advice if needed.

The preprocessor function above can then be used from the :ref:`preprocessors`
like this:

.. code-block:: yaml

   preprocessors:
     example_preprocessor:
       example_preprocessor_function:
         example_argument: median
         example_optional_argument: 6

The optional argument (in this example: ``example_optional_argument``) can be
omitted in the recipe.

Lazy and real data
==================

Preprocessor functions should support both
:ref:`real and lazy data <iris:real_and_lazy_data>`.
This is vital for supporting the large datasets that are typically used with
the ESMValCore.
If the data of the incoming cube has been realized (i.e. ``cube.has_lazy_data()``
returns ``False`` so ``cube.core_data()`` is a `NumPy <https://numpy.org/>`__
array), the returned cube should also have realized data.
Conversely, if the incoming cube has lazy data (i.e. ``cube.has_lazy_data()``
returns ``True`` so ``cube.core_data()`` is a
`Dask array <https://docs.dask.org/en/latest/array.html>`__), the returned
cube should also have lazy data.
Note that NumPy functions will often call their Dask equivalent if it exists
and if their input array is a Dask array, and vice versa.

Note that preprocessor functions should preferably be small and just call the
relevant :ref:`iris <iris_docs>` code.
Code that is more involved, e.g. lots of work with Numpy and Dask arrays,
and more broadly applicable, should be implemented in iris instead.

Documentation
=============

The documentation in the function docstring will be shown in
the :ref:`preprocessor_functions` chapter.
In addition, you should add documentation on how to use the new preprocessor
function from the recipe in
`doc/recipe/preprocessor.rst <https://github.com/ESMValGroup/ESMValCore/tree/main/doc/recipe/preprocessor.rst>`__
so it is shown in the :ref:`preprocessor` chapter.
See the introduction to :ref:`documentation` for more information on how to
best write documentation.

Tests
=====

Tests are should be implemented for new or modified preprocessor functions.
For an introduction to the topic, see :ref:`tests`.

Unit tests
----------

To add a unit test for the preprocessor function from the example above, create
a file called
``tests/unit/preprocessor/_example_module/test_example_preprocessor_function.py``
and add the following content:

.. code-block:: python

    """Test function `esmvalcore.preprocessor.example_preprocessor_function`."""
    import cf_units
    import dask.array as da
    import iris
    import numpy as np
    import pytest

    from esmvalcore.preprocessor import example_preprocessor_function


    @pytest.mark.parametrize('lazy', [True, False])
    def test_example_preprocessor_function(lazy):
        """Test that the computed result is as expected."""

        # Construct the input cube
        data = np.array([1, 2], dtype=np.float32)
        if lazy:
            data = da.asarray(data, chunks=(1, ))
        cube = iris.cube.Cube(
            data,
            var_name='tas',
            units='K',
        )
        cube.add_dim_coord(
            iris.coords.DimCoord(
                np.array([0.5, 1.5], dtype=np.float64),
                bounds=np.array([[0, 1], [1, 2]], dtype=np.float64),
                standard_name='time',
                units=cf_units.Unit('days since 1950-01-01 00:00:00',
                                    calendar='gregorian'),
            ),
            0,
        )

        # Compute the result
        result = example_preprocessor_function(cube, example_argument='time')

        # Check that lazy data is returned if and only if the input is lazy
        assert result.has_lazy_data() is lazy

        # Construct the expected result cube
        expected = iris.cube.Cube(
            np.array(1.5, dtype=np.float32),
            var_name='tas',
            units='K',
        )
        expected.add_aux_coord(
            iris.coords.AuxCoord(
                np.array([1], dtype=np.float64),
                bounds=np.array([[0, 2]], dtype=np.float64),
                standard_name='time',
                units=cf_units.Unit('days since 1950-01-01 00:00:00',
                                    calendar='gregorian'),
            ))
        expected.add_cell_method(
            iris.coords.CellMethod(method='mean', coords=('time', )))

        # Compare the result of the computation with the expected result
        print('result:', result)
        print('expected result:', expected)
        assert result == expected


In this test we used the decorator
`pytest.mark.parametrize <https://docs.pytest.org/en/stable/parametrize.html>`_
to test two scenarios, with both lazy and realized data, with a single test.


Sample data tests
-----------------

The idea of adding :ref:`sample data tests <sample_data_tests>` is to check that
preprocessor functions work with realistic data.
This also provides an easy way to add regression tests, though these should
preferably be implemented as unit tests instead, because using the sample data
for this purpose is slow.
To add a test using the sample data, create a file
``tests/sample_data/preprocessor/example_preprocessor_function/test_example_preprocessor_function.py``
and add the following content:

.. code-block:: python

    """Test function `esmvalcore.preprocessor.example_preprocessor_function`."""
    from pathlib import Path

    import esmvaltool_sample_data
    import iris
    import pytest

    from esmvalcore.preprocessor import example_preprocessor_function


    @pytest.mark.use_sample_data
    def test_example_preprocessor_function():
        """Regression test to check that the computed result is as expected."""
        # Load an example input cube
        cube = esmvaltool_sample_data.load_timeseries_cubes(mip_table='Amon')[0]

        # Compute the result
        result = example_preprocessor_function(cube, example_argument='time')

        filename = Path(__file__).with_name('example_preprocessor_function.nc')
        if not filename.exists():
            # Create the file the expected result if it doesn't exist
            iris.save(result, target=str(filename))
            raise FileNotFoundError(
                f'Reference data was missing, wrote new copy to {filename}')

        # Load the expected result cube
        expected = iris.load_cube(str(filename))

        # Compare the result of the computation with the expected result
        print('result:', result)
        print('expected result:', expected)
        assert result == expected


This will use a file from the sample data repository as input.
The first time you run the test, the computed result will be stored in the file
``tests/sample_data/preprocessor/example_preprocessor_function/example_preprocessor_function.nc``
Any subsequent runs will re-load the data from file and check that it did not
change.
Make sure the stored results are small, i.e. smaller than 100 kilobytes, to
keep the size of the ESMValCore repository small.

Using multiple datasets as input
================================

The name of the first argument of the preprocessor function should in almost all
cases be ``cube``.
Only when implementing a preprocessor function that uses all datasets as input,
the name of the first argument should be ``products``.
If you would like to implement this type of preprocessor function, start by
having a look at the existing functions, e.g.
:py:func:`esmvalcore.preprocessor.multi_model_statistics` or
:py:func:`esmvalcore.preprocessor.mask_fillvalues`.
