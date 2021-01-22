.. _api_recipe_output:

Recipe output
=============

This section describes the :py:mod:`esmvalcore.experimental.recipe_output` submodule of the API.

After running a recipe, output is returned by the :py:meth:`esmvalcore.experimental.recipe.Recipe.run` method. Alternatively, it can be retrieved using the :py:meth:`esmvalcore.experimental.recipe.Recipe.get_output` method.

.. code:: python

    >>> recipe_output = recipe.get_output()

``recipe_output`` is a mapping of the individual tasks and their output
filenames (data and image files) with a set of attributes describing the
data.

.. code:: python

    >>> recipe_output
    timeseries/script1:
      DataFile('tas_amsterdam_CMIP5_CanESM2_Amon_historical_r1i1p1_tas_1850-2000.nc')
      DataFile('tas_amsterdam_CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_1850-2000.nc')
      DataFile('tas_amsterdam_MultiModelMean_Amon_tas_1850-2000.nc')
      DataFile('tas_global_CMIP5_CanESM2_Amon_historical_r1i1p1_tas_1850-2000.nc')
      DataFile('tas_global_CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_1850-2000.nc')
      ImageFile('tas_amsterdam_CMIP5_CanESM2_Amon_historical_r1i1p1_tas_1850-2000.png')
      ImageFile('tas_amsterdam_CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_1850-2000.png')
      ImageFile('tas_amsterdam_MultiModelMean_Amon_tas_1850-2000.png')
      ImageFile('tas_global_CMIP5_CanESM2_Amon_historical_r1i1p1_tas_1850-2000.png')
      ImageFile('tas_global_CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_1850-2000.png')

    map/script1:
      DataFile('CMIP5_CanESM2_Amon_historical_r1i1p1_tas_2000-2000.nc')
      DataFile('CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_2000-2000.nc')
      ImageFile('CMIP5_CanESM2_Amon_historical_r1i1p1_tas_2000-2000.png')
      ImageFile('CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_2000-2000.png')


Output is grouped by the task that produced them. They can be accessed like
a dictionary.

.. code:: python

    >>> task_output = recipe_output['map/script1']
    >>> task_output
    map/script1:
      DataFile('CMIP5_CanESM2_Amon_historical_r1i1p1_tas_2000-2000.nc')
      DataFile('CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_2000-2000.nc')
      ImageFile('CMIP5_CanESM2_Amon_historical_r1i1p1_tas_2000-2000.png')
      ImageFile('CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_2000-2000.png')


The task output has a list of files associated with them, usually image
(``.png``) or data files (``.nc``). To get a list of all files, use
``.files``:

.. code:: python

    >>> print(task_output.files)
    (DataFile('CMIP5_CanESM2_Amon_historical_r1i1p1_tas_2000-2000.nc'),
    ..., ImageFile('CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_2000-2000.png'))


It is also possible to select the image files or data files only:

.. code:: python

    >>> for image_file in task_output.image_files:
    >>>     print(image_file)
    ImageFile('CMIP5_CanESM2_Amon_historical_r1i1p1_tas_2000-2000.png')
    ImageFile('CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_2000-2000.png')

    >>> for data_file in task_output.data_files:
    >>>     print(data_file)
    DataFile('CMIP5_CanESM2_Amon_historical_r1i1p1_tas_2000-2000.nc')
    DataFile('CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_2000-2000.nc')



Working with image files
~~~~~~~~~~~~~~~~~~~~~~~~

Output comes in two kinds, ``DataFile``s, which correspond to data
files in ``.nc`` format and ``ImageFile``s which correspond to plots
in ``.png`` format. Both object are derived from the same base class
(``OutputFile``) and therefore share most of the functionality.

Image output uses IPython magic to plot themselves in a notebook
environment.

.. code:: python

    >>> image_file = recipe_output['map/script1'].image_files[0]
    >>> image_file

For example:

.. image:: /figures/api_recipe_output.png
   :width: 600

Using :py:mod:`IPython.display`, it is possible to show all image files.

.. code:: python

    >>> from IPython.display import display
    >>>
    >>> task = recipe_output['map/script1']
    >>> for image_file in task.image_files:
    >>>      display(image_file)


Working with data output
~~~~~~~~~~~~~~~~~~~~~~~~

Data files can be easily loaded using ``xarray``:

.. code:: python

    >>> data_file = recipe_output['timeseries/script1'].data_files[0]
    >>> data = data_file.load_xarray()
    >>> type(data)
    xarray.core.dataset.Dataset


Or ``iris``:

.. code:: python

    >>> cube = data_file.load_iris()
    >>> type(cube)
    iris.cube.CubeList


API reference
*************

.. automodule:: esmvalcore.experimental.recipe_output
