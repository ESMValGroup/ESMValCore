.. _fixing_data:

***********
Dataset fix
***********

Some (model) datasets contain (known) errors that would normally prevent them
from being processed correctly by the ESMValCore. The errors can be in
the metadata describing the dataset and/or in the actual data.
Typical examples of such errors are missing or wrong attributes (e.g.
attribute ''units'' says 1e-9 but data are actually in 1e-6), missing or
mislabeled coordinates (e.g. ''lev'' instead of ''plev'' or missing
coordinate bounds like ''lat_bnds'') or problems with the actual data
(e.g. cloud liquid water only instead of sum of liquid + ice as specified by the CMIP data request).

The ESMValCore can apply on the fly fixes to datasets that have
known errors that can be fixed automatically.

.. note::
  **CMORization as a fix**.
  Support for many observational and reanalysis datasets is implemented through
  :ref:`CMORizer scripts in the ESMValTool <esmvaltool:new-dataset>`.
  However, it is also possible to add support for a dataset that is not part of
  a CMIP data request by implementing fixes for it.
  This is particularly useful for large datasets, where keeping a copy of both
  the original and CMORized dataset is not feasible.
  See `Natively supported non-CMIP datasets`_ for a list of currently supported
  datasets.


Fix structure
=============

Fixes are Python classes stored in ``esmvalcore/cmor/_fixes/[PROJECT]/[DATASET].py``
that derive from :class:`esmvalcore.cmor._fixes.fix.Fix` and
are named after the short name of the variable they fix. You can use the name
``AllVars`` if you want the fix to be applied to the whole dataset

.. warning::
    Be careful to replace any ``-`` with ``_`` in your dataset name.
    We need this replacement to have proper python module names.

The fixes are automatically loaded and applied when the dataset is preprocessed.
They are a special type of :ref:`preprocessor function <preprocessor_function>`,
called by the preprocessor functions
:py:func:`esmvalcore.preprocessor.fix_file`,
:py:func:`esmvalcore.preprocessor.fix_metadata`, and
:py:func:`esmvalcore.preprocessor.fix_data`.

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
         """Fixes for tas variable.""""

Next we must choose the method to use between the ones offered by the
Fix class:

- ``fix_file`` : should be used only to fix errors that prevent data loading.
  As a rule of thumb, you should only use it if the execution halts before
  reaching the checks.

- ``fix_metadata`` : you want to change something in the cube that is not
  the data (e.g variable or coordinate names, data units).

- ``fix_data``: you need to fix the data. Beware: coordinates data values are
  part of the metadata.

In our case we need to rename the coordinate ``altitude`` to ``latitude``,
so we will implement the ``fix_metadata`` method:

.. code-block:: python

    """Fixes for PERFECT-MODEL."""
    from esmvalcore.cmor.fix import Fix

    class tas(Fix):
        """Fixes for tas variable.""""

        def fix_metadata(self, cubes):
            """
            Fix metadata for tas.

            Fix the name of the latitude coordinate, which is called altitude
            in the original file.
            """"
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
it is a general one, you can change the fix name to ``AllVars`` so it gets
executed for all variables in the dataset. If you find that this is shared only by
a handful of similar vars you can just make the fix for those new vars derive
from the one you just created:

.. code-block:: python

    """Fixes for PERFECT-MODEL."""
    from esmvalcore.cmor.fix import Fix

    class tas(Fix):
        """Fixes for tas variable.""""

        def fix_metadata(self, cubes):
            """
            Fix metadata for tas.

            Fix the name of the latitude coordinate, which is called altitude
            in the original file.
            """"
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
        cube.units = 'real_units'


Detecting this error can be tricky if the units are similar enough. It also
has a good chance of going undetected until you notice strange results in
your diagnostic.

For the above example, it can be useful to access the variable definition
and associated coordinate definitions as provided by the CMOR table.
For example:

.. code-block:: python

    def fix_metadata(self, cubes):
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

.. code-block:: bash

    def fix_metadata(self, cubes):
        coord_cube = cubes.extract_strict('COORDINATE_NAME')
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
        coord.bounds = cubes.extract_strict('BOUNDS_NAME').data

        data_cube = cubes.extract_strict('VAR_NAME')
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


Natively supported non-CMIP datasets
====================================

Fixed datasets are supported through the ``native6`` project.
Put the files containing the data in the directory that you have configured
for the ``native6`` project in your :ref:`user configuration file`, in a
subdirectory called ``Tier{tier}/{dataset}/{version}/{frequency}/{short_name}``.
Replace the items in curly braces by the values used in the variable/dataset
definition in the :ref:`recipe <recipe_overview>`.
Below is a list of datasets currently supported.

ERA5
----

- Supported variables: ``clt``, ``evspsbl``, ``evspsblpot``, ``mrro``, ``pr``, ``prsn``, ``ps``, ``psl``, ``ptype``, ``rls``, ``rlds``, ``rsds``, ``rsdt``, ``rss``, ``uas``, ``vas``, ``tas``, ``tasmax``, ``tasmin``, ``tdps``, ``ts``, ``tsn`` (``E1hr``/``Amon``), ``orog`` (``fx``)
- Tier: 3

MSWEP
-----

- Supported variables: ``pr``
- Supported frequencies: ``mon``, ``day``, ``3hr``.
- Tier: 3

For example for monthly data, place the files in the ``/Tier3/MSWEP/latestversion/mon/pr`` subdirectory of your ``native6`` project location.

.. note::
  For monthly data (V220), the data must be postfixed with the date, i.e. rename ``global_monthly_050deg.nc`` to ``global_monthly_050deg_197901-201710.nc``

For more info: http://www.gloh2o.org/
