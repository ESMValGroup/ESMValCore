.. _preprocessor_function:

Preprocessor function
*********************

Preprocessor functions are located in :py:mod:`esmvalcore.preprocessor`.
To add a new preprocessor function, start by finding a likely looking file to
add your function to in
`esmvalcore/preprocessor <https://github.com/ESMValGroup/ESMValCore/tree/master/esmvalcore/preprocessor>`_.
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
           recipe. Describe what valid values are here.

       example_optional_argument: int, optional
           Another example argument, the value of this argument can optionally
           be provided in the recipe. Describe what valid values are here.

      Returns
      -------
      iris.cube.Cube
          The result of the example computation.
      """

      # Implement your computation here
      return cube


The above function needs to be imported in the file
`esmvalcore/preprocessor/__init__.py <https://github.com/ESMValGroup/ESMValCore/tree/master/esmvalcore/preprocessor/__init__.py>`__:

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

The optional argument can be omitted in the recipe.

Documentation
=============

In addition to the documentation in the function docstring that will be shown in
the :ref:`preprocessor_functions` chapter, add documentation on how to use the
new preprocessor function from the recipe in
`doc/recipe/preprocessor.rst <https://github.com/ESMValGroup/ESMValCore/tree/master/doc/recipe/preprocessor.rst>`__
so it is shown in the :ref:`preprocessor` chapter.
See the introduction to :ref:`documentation` for more information on how to
best write documentation.

Lazy and real data
==================

Preprocessor functions should support both
:ref:`real and lazy data <iris:real_and_lazy_data>`.
This is vital for supporting the large datasets that are typically used with
the ESMValCore.
If the data of the incoming cube has been realised (i.e. ``cube.has_lazy_data()``
returns ``False`` so ``cube.core_data()`` is a `NumPy <https://numpy.org/>`__
array), the returned cube should also have realized data.
Conversely, if the incoming cube has lazy data (i.e. ``cube.has_lazy_data()``
returns ``True`` so ``cube.core_data()`` is a
`Dask array <https://docs.dask.org/en/latest/array.html>`__), the returned
cube should also have lazy data.
Note that NumPy functions will often call their Dask equivalent if it exists
and if their input array is a Dask array, and vice versa.
Preprocessor functions should preferably be small and just call the relevant
:ref:`iris <iris_docs>` code.
Code that is more involved and applicable more broadly than just in the
ESMValCore, should preferably be implemented in iris instead.

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
