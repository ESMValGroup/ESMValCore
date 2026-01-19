.. _cmor_tables:

Variables and CMOR Tables
=========================

ESMValCore has been designed to facilitate working with
`Earth System Model <https://www.climateurope.eu/earth-system-modeling-a-definition/>`__
data, also known as climate model data.
To make it easy to compare and combine data from different climate models,
reanalysis datasets, and observational datasets, ESMValCore uses the standardized
variables from the
`CMOR tables <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables>`_
provided by the projects it supports. `CMOR <https://github.com/PCMDI/cmor>`__
(Climate Model Output Rewriter) is a tool commonly used by climate modelling
centers to format their model output according to community standards.
The CMOR tables define the standardized variable names, units,
coordinates, and other metadata for various climate variables and are typically
compiled from a Data Request and a Controlled Vocabulary, e.g. the
`CMIP7 CMOR tables <https://github.com/WCRP-CMIP/cmip7-cmor-tables/>`__ are
based on the
`CMIP7 Data Request <https://wcrp-cmip.org/cmip-phases/cmip7/cmip7-data-request/>`__,
and the
`CMIP7 Controlled Vocabulary <https://github.com/WCRP-CMIP/CMIP7-CVs>`__
.
ESMValCore comes bundled with several CMOR tables, which are stored in the directory
`esmvalcore/cmor/tables <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables>`_.
It is possible to :ref:`configure which CMOR tables are used by ESMValCore <cmor_table_configuration>`.

The :ref:`facets <facets>` ``project``, ``mip``, ``short_name``, and optionally
``branding_suffix``, uniquely determine the variable to use. These facets are
used to look up the variable in the CMOR table for the project.
Compliance with the variable definition from the CMOR table is checked when data is
loaded, to avoid unexpected results or errors during data processing. The strictness
of these checks can be :ref:`configured <cmor_check_strictness>`.
For example, the facets ``project: CMIP6, mip: Amon, short_name: tas``
define the near-surface air temperature variable in the CMIP6 Amon table:

.. literalinclude:: ../../esmvalcore/cmor/tables/cmip6/Tables/CMIP6_Amon.json
   :start-at: "tas": {
   :end-at: },
   :caption: The ``tas`` variable definition in the CMIP6 Amon table at `esmvalcore/cmor/tables/cmip6/Tables/CMIP6_Amon.json <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables/cmip6/Tables/CMIP6_Amon.json>`__.

In some cases the ``short_name`` (called ``out_name`` in the CMOR tables) of a
variable may differ from the name used as a key in the CMOR table.
This is always the case for CMIP7, where the
`branded variable name <https://wcrp-cmip.github.io/cmip7-guidance/CMIP7/branded_variables/>`__
is used, which is composed of the ``short_name`` followed
by an underscore and the ``branding_suffix``. For example, the facets
``project: CMIP7, mip: atmos, short_name: tas, branding_suffix: tavg-h2m-hxy-u``
select one of the near-surface air temperature variables in the CMIP7 atmos table:

.. literalinclude:: ../../esmvalcore/cmor/tables/cmip7/tables/CMIP7_atmos.json
   :start-at: "tas_tavg-h2m-hxy-u": {
   :end-at: },
   :caption: One of the ``tas`` variable definitions in the CMIP7 atmos table at `esmvalcore/cmor/tables/cmip7/tables/CMIP7_atmos.json <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables/cmip7/Tables/CMIP7_atmos.json>`__.

For other projects, the facet ``branding_suffix`` can also be used to distinguish
between variables from the same CMOR table that share the same ``short_name``,
but differ in other aspects, even though these projects do not use branded variables.
For example, the ``ch4Clim`` entry in the CMIP6 Amon table can be selected in
the recipe by specifying ``project: CMIP6, mip: Amon, short_name: ch4, branding_suffix: Clim``:

.. literalinclude:: ../../esmvalcore/cmor/tables/cmip6/Tables/CMIP6_Amon.json
   :start-at: "ch4Clim": {
   :end-at: },
   :caption: One of the ``ch4`` variable definitions in the CMIP6 Amon table at `esmvalcore/cmor/tables/cmip6/Tables/CMIP6_Amon.json <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables/cmip6/Tables/CMIP6_Amon.json>`__.
