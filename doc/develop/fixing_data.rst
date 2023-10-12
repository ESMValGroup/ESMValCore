.. _fixing_data:

***********
Fixing data
***********

The baseline case for ESMValCore input data is CMOR fully compliant
data that is read using Iris' :func:`iris:iris.load_raw`.
ESMValCore also allows for some departures from compliance (see
:ref:`cmor_check_strictness`). Beyond that situation, some datasets
(either model or observations) contain (known) errors that would
normally prevent them from being processed. The issues can be in
the metadata describing the dataset and/or in the actual data.
Typical examples of such errors are missing or wrong attributes (e.g.
attribute ''units'' says 1e-9 but data are actually in 1e-6), missing or
mislabeled coordinates (e.g. ''lev'' instead of ''plev'' or missing
coordinate bounds like ''lat_bnds'') or problems with the actual data
(e.g. cloud liquid water only instead of sum of liquid + ice as specified by the CMIP data request).
As an extreme case, some data sources simply are not NetCDF
files and must go through some other data load function.
ESMValCore can apply on-the-fly fixes to such datasets when issues can be fixed
automatically.

In addition, some datasets are supported in their native (i.e., non
CMOR-compliant) format through fixes.
This is implemented for a set of :ref:`read_native_datasets`.
A detailed description of how to include new native datasets is given
:ref:`below <add_new_fix_native_datasets>`.

The following sections provide details on how to design such fixes.

.. note::

  **CMORizer scripts**. Support for many observational and reanalysis
  datasets is also possible through a priori reformatting by
  :ref:`CMORizer scripts in the ESMValTool <esmvaltool:new-dataset>`,
  which are rather relevant for datasets of small volume

.. _fix_structure:

Fix structure
=============

Fixes are Python classes stored in
``esmvalcore/cmor/_fixes/[PROJECT]/[DATASET].py`` that derive from
:class:`esmvalcore.cmor._fixes.fix.Fix` and are named after the short name of
the variable they fix. You can also use the names of ``mip`` tables (e.g.,
``Amon``, ``Lmon``, ``Omon``, etc.) if you want the fix to be applied to all
variables of that table in the dataset or ``AllVars`` if you want the fix to be
applied to the whole dataset.

.. warning::
    Be careful to replace any ``-`` with ``_`` in your dataset name.
    We need this replacement to have proper python module names.

The fixes are automatically loaded and applied when the dataset is preprocessed.
They are a special type of :ref:`preprocessor function <preprocessor_function>`,
called by the preprocessor functions
:py:func:`esmvalcore.preprocessor.fix_file`,
:py:func:`esmvalcore.preprocessor.fix_metadata`, and
:py:func:`esmvalcore.preprocessor.fix_data`.

The :class:`~esmvalcore.cmor._fixes.fix.Fix` class provides the following
attributes:

- :attr:`Fix.vardef`: :class:`~esmvalcore.cmor.table.VariableInfo` object that
  corresponds to the variable fixed by the fix.
- :attr:`Fix.extra_facets`: :obj:`dict` that contains all facets of the
  corresponding dataset fixed by the fix (see
  :attr:`esmvalcore.dataset.Dataset.facets`).
- :attr:`Fix.session`: :class:`~esmvalcore.config.Session` object that includes
  configuration and directory information.

Fixing a dataset
================

To illustrate the process of creating a fix we are going to construct a new
one from scratch for a fictional dataset. We need to fix a CMIPX model
called PERFECT-MODEL that is reporting a missing latitude coordinate for
variable tas.

Check the output
----------------

Next to the error message, you should see some info about the iris cube: size,
coordinates. In our example it looks like this:

.. code-block:: python

    air_temperature/ (K) (time: 312; altitude: 90; longitude: 180)
        Dimension coordinates:
            time                                     x              -              -
            altitude                                 -              x              -
            longitude                                -              -              x
        Auxiliary coordinates:
            day_of_month                             x              -              -
            day_of_year                              x              -              -
            month_number                             x              -              -
            year                                     x              -              -
        Attributes:
            {'cmor_table': 'CMIPX', 'mip': 'Amon', 'short_name': 'tas', 'frequency': 'mon'})


So now the mistake is clear: the latitude coordinate is badly named and the
fix should just rename it.

Create the fix
--------------

We start by creating the module file. In our example the path will be
``esmvalcore/cmor/_fixes/CMIPX/PERFECT_MODEL.py``. If it already exists
just add the class to the file, there is no limit in the number of fixes
we can have in any given file.

Then we have to create the class for the fix deriving from
:class:`esmvalcore.cmor._fixes.Fix`

.. code-block:: python

    """Fixes for PERFECT-MODEL."""
    from esmvalcore.cmor.fix import Fix

    class tas(Fix):
         """Fixes for tas variable."""

Next we must choose the method to use between the ones offered by the
Fix class:

- ``fix_file``: should be used only to fix errors that prevent data loading.
  As a rule of thumb, you should only use it if the execution halts before
  reaching the checks.

- ``fix_metadata``: you want to change something in the cube that is not
  the data (e.g., variable or coordinate names, data units).

- ``fix_data``: you need to fix the data. Beware: coordinates data values are
  part of the metadata.

In our case we need to rename the coordinate ``altitude`` to ``latitude``,
so we will implement the ``fix_metadata`` method:

.. code-block:: python

    """Fixes for PERFECT-MODEL."""
    from esmvalcore.cmor.fix import Fix

    class tas(Fix):
        """Fixes for tas variable."""

        def fix_metadata(self, cubes):
            """
            Fix metadata for tas.

            Fix the name of the latitude coordinate, which is called altitude
            in the original file.
            """
            # Sometimes Iris will interpret the data as multiple cubes.
            # Good CMOR datasets will only show one but we support the
            # multiple cubes case to be able to fix the errors that are
            # leading to that extra cubes.
            # In our case this means that we can safely assume that the
            # tas cube is the first one
            tas_cube = cubes[0]
            latitude = tas_cube.coord('altitude')

            # Fix the names. Latitude values, units and
            latitude.short_name = 'lat'
            latitude.standard_name = 'latitude'
            latitude.long_name = 'latitude'
            return cubes

This will fix the error. The next time you run ESMValTool you will find that the error
is fixed on the fly and, hopefully, your recipe will run free of errors.
The ``cubes`` argument to the ``fix_metadata`` method will contain all cubes
loaded from a single input file.
Some care may need to be taken that the right cube is selected and fixed in case
multiple cubes are created.
Usually this happens when a coordinate is mistakenly loaded as a cube, because
the input data does not follow the
`CF Conventions <https://cfconventions.org/>`__.

Sometimes other errors can appear after you fix the first one because they were
hidden by it. In our case, the latitude coordinate could have bad units or
values outside the valid range for example. Just extend your fix to address those
errors.

Finishing
---------

Chances are that you are not the only one that wants to use that dataset and
variable. Other users could take advantage of your fixes as
soon as possible. Please, create a separated pull request for the fix and
submit it.

It will also be very helpful if you just scan a couple of other variables from
the same dataset and check if they share this error. In case that you find that
it is a general one, you can change the fix name to the corresponding ``mip``
table name (e.g., ``Amon``, ``Lmon``, ``Omon``, etc.) so it gets executed for
all variables in that table in the dataset or to ``AllVars`` so it gets
executed for all variables in the dataset. If you find that this is shared only
by a handful of similar vars you can just make the fix for those new vars
derive from the one you just created:

.. code-block:: python

    """Fixes for PERFECT-MODEL."""
    from esmvalcore.cmor.fix import Fix

    class tas(Fix):
        """Fixes for tas variable."""

        def fix_metadata(self, cubes):
            """
            Fix metadata for tas.

            Fix the name of the latitude coordinate, which is called altitude
            in the original file.
            """
            # Sometimes Iris will interpret the data as multiple cubes.
            # Good CMOR datasets will only show one but we support the
            # multiple cubes case to be able to fix the errors that are
            # leading to that extra cubes.
            # In our case this means that we can safely assume that the
            # tas cube is the first one
            tas_cube = cubes[0]
            latitude = tas_cube.coord('altitude')

            # Fix the names. Latitude values, units and
            latitude.short_name = 'lat'
            latitude.standard_name = 'latitude'
            latitude.long_name = 'latitude'
            return cubes


    class ps(tas):
        """Fixes for ps variable."""


Common errors
=============

The above example covers one of the most common cases: variables / coordinates that
have names that do not match the expected. But there are some others that use
to appear frequently. This section describes the most common cases.

Bad units declared
------------------

It is quite common that a variable declares to be using some units but the data
is stored in another. This can be solved by overwriting the units attribute
with the actual data units.

.. code-block:: python

    def fix_metadata(self, cubes):
        cube = self.get_cube_from_list(cubes)
        cube.units = 'real_units'


Detecting this error can be tricky if the units are similar enough. It also
has a good chance of going undetected until you notice strange results in
your diagnostic.

For the above example, it can be useful to access the variable definition
and associated coordinate definitions as provided by the CMOR table.
For example:

.. code-block:: python

    def fix_metadata(self, cubes):
        cube = self.get_cube_from_list(cubes)
        cube.units = self.vardef.units

To learn more about what is available in these definitions, see:
:class:`esmvalcore.cmor.table.VariableInfo` and
:class:`esmvalcore.cmor.table.CoordinateInfo`.



Coordinates missing
-------------------

Another common error is to have missing coordinates. Usually it just means
that the file does not follow the CF-conventions and Iris can therefore not interpret it.

If this is the case, you should see a warning from the ESMValTool about
discarding some cubes in the fix metadata step. Just before that warning you
should see the full list of cubes as read by Iris. If that list contains your
missing coordinate you can create a fix for this model:

.. code-block:: python

    def fix_metadata(self, cubes):
        coord_cube = cubes.extract_cube('COORDINATE_NAME')
        # Usually this will correspond to an auxiliary coordinate
        # because the most common error is to forget adding it to the
        # coordinates attribute
        coord = iris.coords.AuxCoord(
            coord_cube.data,
            var_name=coord_cube.var_name,
            standard_name=coord_cube.standard_name,
            long_name=coord_cube.long_name,
            units=coord_cube.units,
        }

        # It may also have bounds as another cube
        coord.bounds = cubes.extract_cube('BOUNDS_NAME').data

        data_cube = cubes.extract_cube('VAR_NAME')
        data_cube.add_aux_coord(coord, DIMENSIONS_INDEX_TUPLE)
        return [data_cube]


.. _cmor_check_strictness:

Customizing checker strictness
==============================

The data checker classifies its issues using four different levels of
severity. From highest to lowest:

 - ``CRITICAL``: issues that most of the time will have severe consequences.
 - ``ERROR``: issues that usually lead to unexpected errors, but can be safely
   ignored sometimes.
 - ``WARNING``: something is not up to the standard but is unlikely to have
   consequences later.
 - ``DEBUG``: any info that the checker wants to communicate. Regardless of
   checker strictness, those will always be reported as debug messages.

Users can have control about which levels of issues are interpreted as errors,
and therefore make the checker fail or warnings or debug messages.
For this purpose there is an optional command line option `--check-level`
that can take a number of values, listed below from the lowest level of
strictness to the highest:

- ``ignore``: all issues, regardless of severity, will be reported as
  warnings. Checker will never fail. Use this at your own risk.
- ``relaxed``: only CRITICAL issues are treated as errors. We recommend not to
  rely on this mode, although it can be useful if there are errors preventing
  the run that you are sure you can manage on the diagnostics or that will
  not affect you.
- ``default``: fail if there are any CRITICAL or ERROR issues (DEFAULT); Provides
  a good measure of safety.
- ``strict``: fail if there are any warnings, this is the highest level of
  strictness. Mostly useful for checking datasets that you have produced, to
  be sure that future users will not be distracted by inoffensive warnings.


.. _add_new_fix_native_datasets:

Add support for new native datasets
===================================

This section describes how to add support for additional native datasets.
You can choose to host this new data source either under a dedicated project or
under project ``native6``.

.. _add_new_fix_native_datasets_config:

Configuration
-------------

An example of a configuration in ``config-developer.yml`` for projects used for
native datasets is given :ref:`here <configure_native_models>`.
Make sure to use the option ``cmor_strict: false`` for these projects if you
want to make use of :ref:`custom_cmor_tables`.
This allows reading arbitrary variables from native datasets.

.. _add_new_fix_native_datasets_locate_data:

Locate data
-----------

To allow ESMValCore to locate the data files, use the following steps:

   - If you want to use the ``native6`` project (recommended for datasets whose
     input files can be easily moved to the usual ``native6`` directory
     structure given by the ``rootpath`` in your :ref:`user configuration
     file`; this is usually the case for native reanalysis/observational
     datasets):

     The entry ``native6`` of ``config-developer.yml`` should be complemented
     with sub-entries for ``input_dir`` and ``input_file`` that go under a new
     key representing the data organization (such as ``MY_DATA_ORG``), and
     these sub-entries can use an arbitrary list of ``{placeholders}``.
     Example :

     .. code-block:: yaml

        native6:
          ...
          input_dir:
            default: 'Tier{tier}/{dataset}/{version}/{frequency}/{short_name}'
            MY_DATA_ORG: '{dataset}/{exp}/{simulation}/{version}/{type}'
          input_file:
            default: '*.nc'
            MY_DATA_ORG: '{simulation}_*.nc'
          ...

     To find your native data (e.g., called ``MYDATA``) that is for example
     located in ``{rootpath}/MYDATA/amip/run1/42-0/atm/run1_1979.nc``
     (``{rootpath}`` is ESMValTool's ``rootpath`` for the project ``native6``
     defined in your :ref:`user configuration file`), use the following dataset
     entry in your recipe

     .. code-block:: yaml

        datasets:
          - {project: native6, dataset: MYDATA, exp: amip, simulation: run1, version: 42-0, type: atm}

     and make sure to use the following DRS for the project ``native6`` in your
     :ref:`user configuration file`:

     .. code-block:: yaml

        drs:
          native6: MY_DATA_ORG

   - If you want to use a dedicated project for your native dataset
     (recommended for datasets for which you cannot control the location of the
     input files; this is usually the case for native model output):

     A new entry for the project needs to be added to ``config-developer.yml``.
     For example, for the ICON model, create a new project ``ICON``:

     .. code-block:: yaml

        ICON:
          ...
          input_dir:
            default:
              - '{exp}'
              - '{exp}/outdata'
          input_file:
            default: '{exp}_{var_type}*.nc'
          ...

     To find your ICON data that is for example located in files like
     ``{rootpath}/amip/amip_atm_2d_ml_20000101T000000Z.nc`` (``{rootpath}`` is
     ESMValTool ``rootpath`` for the project ``ICON`` defined in your
     :ref:`user configuration file`), use the following dataset entry in your
     recipe:

     .. code-block:: yaml

        datasets:
          - {project: ICON, dataset: ICON, exp: amip}

     Please note the duplication of the name ``ICON`` in ``project`` and
     ``dataset``, which is necessary to comply with ESMValTool's data finding
     and CMORizing functionalities.
     For other native models, ``dataset`` could also refer to a subversion of
     the model.
     Note that it is possible to predefine facets in an :ref:`extra facets file
     <add_new_fix_native_datasets_extra_facets>`.
     In this ICON example, the facet ``var_type`` is :download:`predefined
     </../esmvalcore/config/extra_facets/icon-mappings.yml>` for many
     variables.

.. _add_new_fix_native_datasets_fix_data:

Fix native data
---------------

To ensure that the native dataset has the correct metadata and data (i.e., that
it is CMOR-compliant), use :ref:`dataset fixes <fixing_data>`.
This is where the actual CMORization takes place.
For example, a ``native6`` dataset fix for ERA5 is located `here
<https://github.com/ESMValGroup/ESMValCore/blob/main/esmvalcore/cmor/_fixes/native6/era5.py>`__,
and the ``ICON`` fix is located `here
<https://github.com/ESMValGroup/ESMValCore/blob/main/esmvalcore/cmor/_fixes/icon/icon.py>`__.

ESMValTool also provides a base class ``NativeDatasetFix`` that provides
convenient functions useful for all native dataset fixes.
An example for its usage can be found `here
<https://github.com/ESMValGroup/ESMValCore/blob/main/esmvalcore/cmor/_fixes/icon/_base_fixes.py>`__.

.. _add_new_fix_native_datasets_extra_facets:

Extra facets for native datasets
--------------------------------

If necessary, provide a so-called ``extra facets file`` which allows to cope
e.g. with variable naming issues for finding files or additional information
that is required for the fixes.
See :ref:`extra_facets` and :ref:`extra-facets-fixes` for more details on this.
An example of such a file for IPSL-CM6 is given :download:`here
<../../esmvalcore/config/extra_facets/ipslcm-mappings.yml>`.


.. _extra-facets-fixes:

Use of extra facets in fixes
============================
Extra facets are a mechanism to provide additional information for certain kinds
of data. The general approach is described in :ref:`extra_facets`. Here, we
describe how they can be used in fixes to mold data into the form required by
the applicable standard. For example, if the input data is part of an
observational product that delivers surface temperature with a variable name of
`t2m` inside a file named `2m_temperature_1950_monthly.nc`, but the same
variable is called `tas` in the applicable standard, a fix can be created that
reads the original variable from the correct file, and provides a renamed
variable to the rest of the processing chain.

Normally, the applicable standard for variables is CMIP6.

For more details, refer to existing uses of this feature as examples,
as e.g. :ref:`for IPSL-CM6<ipslcm_extra_facets_example>`.
