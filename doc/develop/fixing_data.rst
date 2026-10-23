.. _fixing_data:

***********
Fixing data
***********

The baseline case for ESMValCore input data is fully :ref:`CMOR compliant <cmor_tables>`
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

Fixes can be written in two ways:

- **With woodpecker (recommended).**
  `Woodpecker <https://github.com/roocs/woodpecker>`__ fixes data in
  :class:`xarray.Dataset` form before ESMValCore converts it to Iris cubes.
  Fixes are combined into recipes, which can be distributed in plugins and
  shared with other tools, or kept in a user's own recipe directory without
  changing ESMValCore.
  New fixes should be written for woodpecker, see :ref:`woodpecker_fixes`.
- **With the fixes included in ESMValCore (legacy).**
  These are Python classes in the ESMValCore source code, described from
  :ref:`fix_structure` onwards.
  They are still applied to all data that no woodpecker recipe matches, but
  they are considered legacy: they will be ported to woodpecker over time,
  and new fixes should not be added to ESMValCore.

.. note::

  **CMORizer scripts**. Support for many observational and reanalysis
  datasets is also possible through a priori reformatting by
  :ref:`CMORizer scripts in the ESMValTool <esmvaltool:new-dataset>`,
  which are rather relevant for datasets of small volume

.. _woodpecker_fixes:

Fixing data with woodpecker
===========================

How ESMValCore applies woodpecker recipes
-----------------------------------------

ESMValCore looks for a woodpecker recipe that matches the data when it loads

- local netCDF files (``.nc`` and ``.nc4``), including files downloaded from
  ESGF, and
- data found with :mod:`esmvalcore.io.intake_esgf`.

Recipes match on the global attributes of the data and on its name.
For local files the name is the full path of the file, for
:mod:`esmvalcore.io.intake_esgf` it is the ESGF dataset id including the
version, e.g.
``CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.tas.gn.v20190429``.
To match all versions of a dataset, end the pattern with ``*``.
At most one recipe may match: if more than one matches, ESMValCore stops with
an error.

If a recipe matches, the data is fixed by woodpecker only:

1. ESMValCore opens the data with :mod:`xarray`. Local files are opened with
   ``decode_times=False``.
2. The data is masked as Iris would mask it when loading the file: values
   equal to ``_FillValue``, ``missing_value`` or the netCDF default fill value,
   and values outside ``valid_range``, ``valid_min`` and ``valid_max``, become
   NaN. The range attributes that have been applied are removed.
3. Woodpecker applies the recipe.
4. The data is converted to Iris cubes. NaN values are masked.
5. ESMValCore does not apply any of its own fixes to the data, not even the
   generic ones. It recognizes the data by the global attribute
   ``woodpecker_provenance``, which records the recipe and the fixes that
   changed the data. Files fixed earlier with the woodpecker command line
   tool and saved with this attribute are recognized in the same way.
6. The data is checked against the CMOR tables as usual, if the project has
   CMOR tables.

If no recipe matches, the data is loaded and fixed by ESMValCore as before.

All files of a dataset should be fixed in the same way.
If only some files of a dataset match a recipe, ESMValCore logs a warning:
after the files have been concatenated, the data of the whole dataset is
treated as fixed by woodpecker, so the data fixes of ESMValCore are not
applied to the other files.

To see which recipe is applied to which file, and which fixes changed the
data, run ESMValCore with ``log_level: debug``.

Adding recipes
--------------

Woodpecker looks for recipes in

1. the files and directories listed in the environment variable
   ``WOODPECKER_RECIPE_PATH``, separated by ``:``,
2. the user directories ``~/.config/woodpecker/recipes`` and
   ``~/.woodpecker/recipes``,
3. the system directories ``/etc/woodpecker/recipes`` and
   ``/etc/woodpecker``,
4. woodpecker itself, and
5. the installed woodpecker plugins.

In a directory, all ``.yaml``, ``.yml`` and ``.json`` files are read;
subdirectories are not searched.
If several recipes have the same ``id``, the first one found is used, so a
user can replace a recipe from a plugin by writing a recipe with the same id.
A recipe with a different id that matches the same data does not replace it:
both recipes match and ESMValCore stops with an error.

A recipe consists of a ``match`` section that says which data it applies to,
and the ``steps``, the fixes to apply in order, each with its ``options``.
For example, the ocean files of a model may store the latitude and longitude
of its curvilinear grid as ordinary data variables ``nav_lat`` and
``nav_lon``, without listing them in the ``coordinates`` attribute of the
data variable.
Iris then loads them as separate cubes, so the data has no latitude and
longitude.
This recipe marks them as coordinates, so they become auxiliary coordinates
of the data:

.. code-block:: yaml

    recipes:
      - id: my_project.my_model_ocean
        description: Fix the ocean grid of MY-MODEL
        match:
          path_patterns:
            - "*/MY-MODEL/Omon/*.nc"
        steps:
          - id: woodpecker.set_coordinate_variables
            options:
              coordinates:
                - nav_lat
                - nav_lon

Instead of ``path_patterns``, which are matched against the name of the data
with :func:`fnmatch.fnmatch`, a recipe can match on global attributes with
``attrs`` (e.g. ``source_id: MY-MODEL``) or on the ``dataset_id`` or
``ds_id`` global attribute with ``dataset_id_patterns``.
If several of these are given, all of them must match.

Recipes can use the fixes included in woodpecker and the fixes of the
installed plugins.
The available fixes are listed in the
`woodpecker fix reference <https://roocs.github.io/woodpecker/FIXES/>`__.

Writing new fixes
-----------------

A fix that cannot be built from existing fixes is written in Python and
distributed in a woodpecker plugin, a Python package that registers its fixes
when it is imported.
Woodpecker finds plugins through the ``woodpecker.plugins`` entry point, so
the package needs to be installed, e.g. with ``pip install --editable .``
for a plugin that is still being developed.
Recipes in the ``recipes`` directory of the package named in the entry point
are found automatically.

A fix is a class registered with
``woodpecker.fixes.registry.register_fix_function``.
Its id is ``<prefix>.<suffix>``.
The ``apply`` method changes the dataset in place and returns whether it
changed anything; with ``dry_run=True`` it must not change the dataset.
The options from the recipe are available as ``self.config``.
For example, a fix that flips the sign of the variables given in the
options:

.. code-block:: python

    """Woodpecker fixes for MY-MODEL."""

    import xarray as xr
    from woodpecker.fixes.labels import Labels
    from woodpecker.fixes.registry import FixFunction, register_fix_function


    @register_fix_function
    class FlipSign(FixFunction):
        """Flip the sign of the variables given in the recipe options."""

        prefix = "my_model"
        suffix = "flip_sign"
        name = "Flip sign"
        description = "Multiplies the configured variables by -1."
        categories = ["data"]
        priority = 50
        labels = [Labels.RISK_VALUE_TRANSFORMATION]

        def _variables(self, dataset: xr.Dataset) -> list[str]:
            config = getattr(self, "config", {}) or {}
            return [v for v in config.get("variables", []) if v in dataset]

        def matches(self, dataset: xr.Dataset) -> bool:
            return bool(self._variables(dataset))

        def apply(self, dataset: xr.Dataset, dry_run: bool = True) -> bool:
            variables = self._variables(dataset)
            if variables and not dry_run:
                for name in variables:
                    attrs = dataset[name].attrs
                    encoding = dataset[name].encoding
                    dataset[name] = -dataset[name]
                    dataset[name].attrs = attrs
                    dataset[name].encoding = encoding
            return bool(variables)

and the entry point in the ``pyproject.toml`` file of the plugin:

.. code-block:: toml

    [project.entry-points."woodpecker.plugins"]
    my_model = "my_woodpecker_plugin"

A recipe then uses the fix with ``id: my_model.flip_sign`` and
``options: {variables: [hfls]}``.

Guidelines for recipes used with ESMValCore
-------------------------------------------

- **Produce CMOR-compliant data.**
  ESMValCore applies none of its own fixes to data fixed by woodpecker, not
  even the generic ones that, e.g., fix units, coordinate names and bounds,
  or the direction of coordinates. The recipe needs to take care of
  everything that the CMOR checks would otherwise complain about.
- **Support encoded times.**
  Local files are opened with ``decode_times=False``, because Iris needs the
  encoded times, while data from :mod:`esmvalcore.io.intake_esgf` has decoded
  times. Fixes that work on the time coordinate should support both.
- **Use NaN for missing values and keep the encoding.**
  Missing values arrive as NaN, and any NaN left after the fixes is masked.
  Fixes should keep the ``encoding`` of the variables they change where
  possible, in particular that of integer variables such as the
  connectivities of UGRID meshes: without its ``_FillValue``, a connectivity
  with missing indices is converted to floating point numbers and Iris no
  longer recognizes the mesh.
- **Match all files of a dataset.**
  A recipe should match all files of a dataset or none, see above.
- **Match each dataset with one recipe only.**
  Give recipes specific ``match`` sections, so that they do not match data
  that another recipe also matches.

.. _fix_structure:

Fix structure
=============

.. note::

    This section and the following ones describe the fixes included in
    ESMValCore. These are considered legacy: new fixes should be written for
    woodpecker, see :ref:`woodpecker_fixes`.

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

- ``fix_file``: you need to fix errors that prevent loading the data with Iris
  or perform operations that are more efficient with other packages (e.g.,
  loading files with lots of variables is much faster with Xarray than Iris).

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

The baseline case for ESMValCore input data is fully
:ref:`CMOR compliant <cmor_tables>` data and this is checked when data is
loaded by :meth:`esmvalcore.dataset.Dataset.load`.

However, it is possible to disable these checks completely by
:ref:`configuring the project <cmor_table_configuration>` so it uses the
:class:`esmvalcore.cmor.table.NoInfo` CMOR table,
or to adjust the strictness of the checks using the :ref:`configuration option
<config_options>` ``check_level``.

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
For this purpose there is an optional :ref:`configuration option
<config_options>` ``check_level`` that can take a number of values, listed
below from the lowest level of strictness to the highest:

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

.. warning::

    While it is possible to work with datasets that are not described in a CMOR
    table or only partially follow the CMOR standards, the
    :ref:`preprocessor functions <preprocessor>` and
    :ref:`diagnostics <esmvaltool:recipes>` have been designed to work with
    CMORized data and may not work as expected with non-CMORized data.

    Our ambition is that :ref:`preprocessor functions <preprocessor>` support
    data that follows the `CF Conventions <https://cfconventions.org/>`_.

.. _add_new_fix_native_datasets:

Add support for new native datasets
===================================

This section describes how to add support for additional native datasets.
You can choose to host this new data source either under a dedicated project or
under project ``native6``.

.. _add_new_fix_native_datasets_config:

CMOR Table Configuration
------------------------

An example of a CMOR table configuration for projects used for native datasets is given here:

.. literalinclude:: ../configurations/defaults/cmor_tables.yml
    :language: yaml
    :caption: Example native data format projects in ``defaults/cmor_tables.yml``
    :prepend: projects:
    :start-at: # Observational and reanalysis data that can be read in its native format by ESMValCore.
    :end-before: CESM:

The option ``strict: false`` is convenient for these projects if you
want to make use of the feature that looks in all tables instead of
only the one specified by the ``mip`` facet in the :ref:`recipe <recipe>` or
:class:`~esmvalcore.dataset.Dataset`: like this, a custom variable only needs to
be defined for a single ``mip`` and can then be used with all ``mip`` s.

.. _add_new_fix_native_datasets_locate_data:

Locate data
-----------

To allow ESMValCore to locate the data files, use the following steps:

- If you want to use the ``native6`` project, recommended for datasets whose
  input files can be easily moved to the usual ``native6`` directory
  structure given by

  .. literalinclude:: ../configurations/data-local-esmvaltool.yml
     :language: yaml
     :caption: ``native6`` standard directory organization in ``data-local-esmvaltool.yml``
     :end-before: # Data that has been CMORized by ESMValTool according to the CMIP6 standard.

  this is preferred.

- If you want to use a dedicated project for your native dataset
  (this is usually the case for native model output):

  A new entry for the project needs to be added under :ref:`config-projects`.
  For example, for the ICON model, create a new project ``ICON`` and define
  its data sources:

  .. literalinclude:: ../configurations/data-native-icon.yml
     :language: yaml
     :caption: ``ICON`` standard directory organization in ``data-native-icon.yml``

  and a CMOR table configuration:

  .. literalinclude:: ../configurations/defaults/cmor_tables.yml
     :language: yaml
     :caption: ``ICON`` CMOR table configuration from ``defaults/cmor_tables.yml``
     :prepend: projects:
     :start-at: ICON:
     :end-at: strict: false

  To find your ICON data that is, for example, located in files like
  ``~/climate_data/amip/amip_atm_2d_ml_20000101T000000Z.nc``, use the following
  dataset entry in your recipe:

  .. code-block:: yaml

     datasets:
       - {project: ICON, dataset: ICON, exp: amip}

  Please note the duplication of the name ``ICON`` in ``project`` and
  ``dataset``, which is necessary to comply with ESMValTool's data finding
  and CMORizing functionalities.
  For other native models, ``dataset`` could also refer to a subversion of
  the model.
  Note that it is possible to predefine facets via :ref:`extra facets
  <add_new_fix_native_datasets_extra_facets>`.
  In this ICON example, the facet ``output_stream`` is :download:`predefined
  </../esmvalcore/config/configurations/defaults/extra_facets_icon.yml>`
  for many variables.

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

If necessary, so-called :ref:`config-extra-facets` can be provided which allow
to cope e.g. with variable naming issues for finding files or additional
information that is required for the fixes.
See :ref:`extra-facets-fixes` for more details on this.
An example of such a configuration file for IPSL-CM6 is given :download:`here
<../../esmvalcore/config/configurations/defaults/extra_facets_ipslcm.yml>`.


.. _extra-facets-fixes:

Use of extra facets in fixes
============================
Extra facets are a mechanism to provide additional information for certain
kinds of data. The general approach is described in :ref:`config-extra-facets`.
Here, we describe how they can be used in fixes to mold data into the form
required by the applicable standard. For example, if the input data is part of
an observational product that delivers surface temperature with a variable name
of ``t2m`` inside a file named ``2m_temperature_1950_monthly.nc``, but the same
variable is called ``tas`` in the applicable standard, a fix can be created
that reads the original variable from the correct file, and provides a renamed
variable to the rest of the processing chain.

Normally, the applicable standard for variables is CMIP6.

For more details, refer to existing uses of this feature as examples,
as e.g. :ref:`for IPSL-CM6<ipslcm_extra_facets_example>`.
