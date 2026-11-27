.. _api_recipe_output:

Recipe output
=============

This section describes the :py:mod:`~esmvalcore.experimental.recipe_output` submodule of the API (:py:mod:`esmvalcore.experimental`).

After running a recipe, output is returned by the :py:meth:`~esmvalcore.experimental.recipe.Recipe.run` method. Alternatively, it can be retrieved using the :py:meth:`~esmvalcore.experimental.recipe.Recipe.get_output` method.

.. code:: python

    >>> recipe_output = recipe.get_output()

``recipe_output`` is a mapping of the individual tasks and their output
filenames with a set of attributes describing each file.

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


Each task output has an associated list of files.
To get a list of all files, use
:py:meth:`~esmvalcore.experimental.recipe_output.TaskOutput.files`.

.. code:: python

    >>> print(task_output.files)
    (DataFile('CMIP5_CanESM2_Amon_historical_r1i1p1_tas_2000-2000.nc'),
    ..., ImageFile('CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_2000-2000.png'))


Use the following properties to select files by type.

- :py:attr:`~esmvalcore.experimental.recipe_output.TaskOutput.image_files`
- :py:attr:`~esmvalcore.experimental.recipe_output.TaskOutput.data_files`
- :py:attr:`~esmvalcore.experimental.recipe_output.TaskOutput.table_files`

.. code:: python

    >>> for image_file in task_output.image_files:
    >>>     print(image_file)
    ImageFile('CMIP5_CanESM2_Amon_historical_r1i1p1_tas_2000-2000.png')
    ImageFile('CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_2000-2000.png')

    >>> for data_file in task_output.data_files:
    >>>     print(data_file)
    DataFile('CMIP5_CanESM2_Amon_historical_r1i1p1_tas_2000-2000.nc')
    DataFile('CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_2000-2000.nc')


Working with output files
*************************

Output files use subclasses of
:py:class:`~esmvalcore.experimental.recipe_output.OutputFile`.

- :py:class:`~esmvalcore.experimental.recipe_output.DataFile` for NetCDF data
- :py:class:`~esmvalcore.experimental.recipe_output.ImageFile` for PNG plots
- :py:class:`~esmvalcore.experimental.recipe_output.TableFile` for CSV tables

For example, author information can be accessed as instances of :py:class:`~esmvalcore.experimental.recipe_metadata.Contributor`  via

.. code:: python

    >>> output_file = task_output[0]
    >>> output_file.authors
    (Contributor('Andela, Bouwe', institute='NLeSC, Netherlands', orcid='https://orcid.org/0000-0001-9005-8940'),
     Contributor('Righi, Mattia', institute='DLR, Germany', orcid='https://orcid.org/0000-0003-3827-5950'))

And associated references as instances of :py:class:`~esmvalcore.experimental.recipe_metadata.Reference` via

.. code:: python

    >>> output_file.references
    (Reference('acknow_project'),)

:py:class:`~esmvalcore.experimental.recipe_output.OutputFile` also knows about associated files

.. code:: python

    >>> data_file.citation_file
    Path('.../tas_global_CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_1850-2000_citation.bibtex')
    >>> data_file.data_citation_file
    Path('.../tas_global_CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_1850-2000_data_citation_info.txt')
    >>> data_file.provenance_svg_file
    Path('.../tas_global_CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_1850-2000_provenance.svg')
    >>> data_file.provenance_xml_file
    Path('.../tas_global_CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_1850-2000_provenance.xml')



Working with image files
************************

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


Working with data files
***********************

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
