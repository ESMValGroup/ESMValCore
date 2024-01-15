.. _changelog:

Changelog
=========

.. _changelog-v2-10-0:

v2.10.0
-------
Highlights

TODO: add highlights

This release includes

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Remove the deprecated option ``use_legacy_supplementaries`` (`#2202 <https://github.com/ESMValGroup/ESMValCore/pull/2202>`__) `Bouwe Andela <https://github.com/bouweandela>`__

   - The recommended upgrade procedure is to remove ``use_legacy_supplementaries`` from config-user.yml
     (if it was there) and remove any mention of ``fx_variables`` from the recipe. If automatically defining
     the required supplementary variables does not work, define them in the variable or
     (``additional_``) ``datasets`` section as described in :ref:`supplementary_variables`.

-  Use smarter (units-aware) weights (`#2139 <https://github.com/ESMValGroup/ESMValCore/pull/2139>`__) `Manuel Schlund <https://github.com/schlunma>`__
   
   - Some preprocessors handle units better. For details, see the pull request.

-  Removed deprecated configuration option ``offline`` (`#2213 <https://github.com/ESMValGroup/ESMValCore/pull/2213>`__) `Manuel Schlund <https://github.com/schlunma>`__

   - In :ref:`changelog-v2-8-0`, we replaced the old ``offline`` configuration option. From this version on, it stops working.
     Please refer to :ref:`changelog-v2-8-0` for upgrade instructions.

-  Fix issue with CORDEX datasets requiring different dataset tags for downloads and fixes (`#2066 <https://github.com/ESMValGroup/ESMValCore/pull/2066>`__) `Joakim Löw <https://github.com/ljoakim>`__
   
   - Due to the different facets for CORDEX datasets, there was an inconsistency in the fixing mechanism.
     This change requires changes to existing recipes that use CORDEX datasets. Please refer to the pull request for detailed update instructions.

-  Added new operators for statistics preprocessor (e.g., ``'percentile'``) and allowed passing additional arguments (`#2191 <https://github.com/ESMValGroup/ESMValCore/pull/2191>`__) `Manuel Schlund <https://github.com/schlunma>`__
   
   - This harmonizes the operators for all statistics preprocessors. From this version, the new names can be used; the old arguments will stop working from
     version 2.12.0. Please refer to :ref:`stat_preprocs` for a detailed description.

-  For the following changes, no user change is necessary
  
   -  Remove deprecated way of calling :func:`~esmvalcore.cmor.table.read_cmor_tables` (`#2201 <https://github.com/ESMValGroup/ESMValCore/pull/2201>`__) `Bouwe Andela <https://github.com/bouweandela>`__

   -  Remove deprecated callback argument from preprocessor ``load`` function (`#2207 <https://github.com/ESMValGroup/ESMValCore/pull/2207>`__) `Bouwe Andela <https://github.com/bouweandela>`__

   -  Remove deprecated preprocessor function `cleanup` (`#2215 <https://github.com/ESMValGroup/ESMValCore/pull/2215>`__) `Bouwe Andela <https://github.com/bouweandela>`__


Deprecations
~~~~~~~~~~~~

-  Clearly separate fixes and CMOR checks (`#2157 <https://github.com/ESMValGroup/ESMValCore/pull/2157>`__) `Manuel Schlund <https://github.com/schlunma>`__

Bug fixes
~~~~~~~~~

-  Re-add correctly region-extracted cell measures and ancillary variables after :ref:`extract_region` (`#2166 <https://github.com/ESMValGroup/ESMValCore/pull/2166>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__, `Manuel Schlund <https://github.com/schlunma>`__
-  Fix sorting of datasets

   -  Fix sorting of ensemble members in :func:`~esmvalcore.dataset.datasets_to_recipe` (`#2095 <https://github.com/ESMValGroup/ESMValCore/pull/2095>`__) `Bouwe Andela <https://github.com/bouweandela>`__
   -  Fix a problem with sorting datasets that have a mix of facet types (`#2238 <https://github.com/ESMValGroup/ESMValCore/pull/2238>`__) `Bouwe Andela <https://github.com/bouweandela>`__
   -  Avoid a crash if dataset has supplementary variables (`#2198 <https://github.com/ESMValGroup/ESMValCore/pull/2198>`__) `Bouwe Andela <https://github.com/bouweandela>`__

CMOR standard
~~~~~~~~~~~~~

-  ERA5 on-the-fly CMORizer: changed sign of variables ``evspsbl`` and ``evspsblpot`` (`#2115 <https://github.com/ESMValGroup/ESMValCore/pull/2115>`__) `katjaweigel <https://github.com/katjaweigel>`__
-  Add ``ch4`` surface custom cmor table entry (`#2168 <https://github.com/ESMValGroup/ESMValCore/pull/2168>`__) `Birgit Hassler <https://github.com/hb326>`__
-  Add CMIP3 institutes names used at NCI (`#2152 <https://github.com/ESMValGroup/ESMValCore/pull/2152>`__) `Romain Beucher <https://github.com/rbeucher>`__
-  Added :func:`~esmvalcore.cmor.fixes.get_time_bounds` and :func:`~esmvalcore.cmor.fixes.get_next_month` to public API (`#2214 <https://github.com/ESMValGroup/ESMValCore/pull/2214>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Improve concatenation checks

   -  Relax concatenation checks for ``--check_level=relax`` and ``--check_level=ignore`` (`#2144 <https://github.com/ESMValGroup/ESMValCore/pull/2144>`__) `sloosvel <https://github.com/sloosvel>`__
   -  Fix ``concatenate`` preprocessor function (`#2240 <https://github.com/ESMValGroup/ESMValCore/pull/2240>`__) `Bouwe Andela <https://github.com/bouweandela>`__
   -  Fix time overlap handling in concatenation (`#2247 <https://github.com/ESMValGroup/ESMValCore/pull/2247>`__) `Klaus Zimmermann <https://github.com/zklaus>`__

Computational performance improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Make :ref:`threshold_masking` preprocessors lazy  (`#2169 <https://github.com/ESMValGroup/ESMValCore/pull/2169>`__) `Jörg Benke <https://github.com/joergbenke>`__

   -  Restored usage of numpy in `_mask_with_shp` (`#2209 <https://github.com/ESMValGroup/ESMValCore/pull/2209>`__) `Jörg Benke <https://github.com/joergbenke>`__
-  Call coord.core_bounds() instead of coord.bounds in ``check.py`` (`#2146 <https://github.com/ESMValGroup/ESMValCore/pull/2146>`__) `sloosvel <https://github.com/sloosvel>`__
-  Rechunk between preprocessor steps (`#2205 <https://github.com/ESMValGroup/ESMValCore/pull/2205>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Reduce the size of the dask graph created by the ``anomalies`` preprocessor function (`#2200 <https://github.com/ESMValGroup/ESMValCore/pull/2200>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Documentation
~~~~~~~~~~~~~

-  Add reference to release v2.9.0 in the changelog (`#2130 <https://github.com/ESMValGroup/ESMValCore/pull/2130>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Add merge instructions to release instructions (`#2131 <https://github.com/ESMValGroup/ESMValCore/pull/2131>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Update `mamba` before building environment during Readthedocs build (`#2149 <https://github.com/ESMValGroup/ESMValCore/pull/2149>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Ensure compatible zstandard and zstd versions for .conda support (`#2204 <https://github.com/ESMValGroup/ESMValCore/pull/2204>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Remove outdated documentation (`#2210 <https://github.com/ESMValGroup/ESMValCore/pull/2210>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Remove meercode badge from README because their API is broken (`#2224 <https://github.com/ESMValGroup/ESMValCore/pull/2224>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Correct usage help text of version command (`#2232 <https://github.com/ESMValGroup/ESMValCore/pull/2232>`__) `James Frost <https://github.com/jfrost-mo>`__
-  Add ``navigation_with_keys: False`` to ``html_theme_options`` in Readthedocs ``conf.py`` (`#2245 <https://github.com/ESMValGroup/ESMValCore/pull/2245>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Replace squarey badge with roundy shield for Anaconda sticker in README (`#2233 <https://github.com/ESMValGroup/ESMValCore/pull/2233>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Updated doc about fixes and added type hints to fix functions (`#2160 <https://github.com/ESMValGroup/ESMValCore/pull/2160>`__) `Manuel Schlund <https://github.com/schlunma>`__

Installation
~~~~~~~~~~~~

-  Clean-up how pins are written in conda environment file (`#2125 <https://github.com/ESMValGroup/ESMValCore/pull/2125>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Use importlib.metadata instead of deprecated pkg_resources (`#2096 <https://github.com/ESMValGroup/ESMValCore/pull/2096>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Pin shapely to >=2.0 (`#2075 <https://github.com/ESMValGroup/ESMValCore/pull/2075>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

Preprocessor
~~~~~~~~~~~~

-  Improve preprocessor output sorting code (`#2111 <https://github.com/ESMValGroup/ESMValCore/pull/2111>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Preprocess datasets in the same order as they are listed in the recipe (`#2103 <https://github.com/ESMValGroup/ESMValCore/pull/2103>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Automatic testing
~~~~~~~~~~~~~~~~~

-  [Github Actions] Compress all bash shell setters into one default option per workflow (`#2126 <https://github.com/ESMValGroup/ESMValCore/pull/2126>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  [Github Actions] Fix Monitor Tests Github Action (`#2135 <https://github.com/ESMValGroup/ESMValCore/pull/2135>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  [condalock] update conda lock file (`#2141 <https://github.com/ESMValGroup/ESMValCore/pull/2141>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  [Condalock] make sure mamba/conda are at latest version by forcing a pinned mamba install (`#2136 <https://github.com/ESMValGroup/ESMValCore/pull/2136>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Update code coverage orbs (`#2206 <https://github.com/ESMValGroup/ESMValCore/pull/2206>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Revisit the comment-triggered Github Actions test (`#2243 <https://github.com/ESMValGroup/ESMValCore/pull/2243>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Remove workflow that runs Github Actions tests from PR comment (`#2244 <https://github.com/ESMValGroup/ESMValCore/pull/2244>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

Improvements
~~~~~~~~~~~~

-  Merge v2.9.x into main (`#2128 <https://github.com/ESMValGroup/ESMValCore/pull/2128>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fix typo in citation file (`#2182 <https://github.com/ESMValGroup/ESMValCore/pull/2182>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Cleaned and extended function that extracts datetimes from paths (`#2181 <https://github.com/ESMValGroup/ESMValCore/pull/2181>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Add file encoding (and some read modes) at open file step (`#2219 <https://github.com/ESMValGroup/ESMValCore/pull/2219>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Check type of argument passed to :func:`~esmvalcore.cmor.table.read_cmor_tables` (`#2217 <https://github.com/ESMValGroup/ESMValCore/pull/2217>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Dynamic HTML output for monitoring (`#2062 <https://github.com/ESMValGroup/ESMValCore/pull/2062>`__) `Brei Soliño <https://github.com/bsolino>`__


.. _changelog-v2-9-0:


v2.9.0
------
Highlights
~~~~~~~~~~
It is now possible to use the
`Dask distributed scheduler <https://docs.dask.org/en/latest/deploying.html>`__,
which can
`significantly reduce the run-time of recipes <https://github.com/ESMValGroup/ESMValCore/pull/2049#pullrequestreview-1446279391>`__.
Configuration examples and advice are available in
:ref:`our documentation <config-dask>`.
More work on improving the computational performance is planned, so please share
your experiences, good and bad, with this new feature in
`ESMValGroup/ESMValCore#1763 <https://github.com/ESMValGroup/ESMValCore/discussions/1763>`__.

This release includes

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Remove deprecated configuration options (`#2056 <https://github.com/ESMValGroup/ESMValCore/pull/2056>`__) `Bouwe Andela <https://github.com/bouweandela>`__

   - The module ``esmvalcore.experimental.config`` has been removed.
     To upgrade, import the module from :mod:`esmvalcore.config` instead.

   - The module ``esmvalcore._config`` has been removed.
     To upgrade, use :mod:`esmvalcore.config` instead.

   - The methods ``esmvalcore.config.Session.to_config_user`` and ``esmvalcore.config.Session.from_config_user`` have been removed.
     To upgrade, use :obj:`esmvalcore.config.Session` to access the configuration values directly.

Bug fixes
~~~~~~~~~

-  Respect ``ignore_warnings`` settings from the :ref:`project configuration <filterwarnings_config-developer>` in :func:`esmvalcore.dataset.Dataset.load` (`#2046 <https://github.com/ESMValGroup/ESMValCore/pull/2046>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fixed usage of custom location for :ref:`custom CMOR tables <custom_cmor_tables>` (`#2052 <https://github.com/ESMValGroup/ESMValCore/pull/2052>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fix issue with writing index.html when :ref:`running a recipe <running>` with ``--resume-from`` (`#2055 <https://github.com/ESMValGroup/ESMValCore/pull/2055>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fixed bug in ICON CMORizer that lead to shifted time coordinates (`#2038 <https://github.com/ESMValGroup/ESMValCore/pull/2038>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Include ``-`` in allowed characters for bibtex references (`#2097 <https://github.com/ESMValGroup/ESMValCore/pull/2097>`__) `Alistair Sellar <https://github.com/alistairsellar>`__
-  Do not raise an exception if the requested version of a file is not available for all matching files on ESGF (`#2105 <https://github.com/ESMValGroup/ESMValCore/pull/2105>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Computational performance improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Add support for :ref:`configuring Dask distributed <config-dask>` (`#2049 <https://github.com/ESMValGroup/ESMValCore/pull/2049>`__, `#2122 <https://github.com/ESMValGroup/ESMValCore/pull/2122>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Make :func:`esmvalcore.preprocessor.extract_levels` lazy (`#1761 <https://github.com/ESMValGroup/ESMValCore/pull/1761>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Lazy implementation of :func:`esmvalcore.preprocessor.multi_model_statistics` and :func:`esmvalcore.preprocessor.ensemble_statistics` (`#968 <https://github.com/ESMValGroup/ESMValCore/pull/968>`__ and `#2087 <https://github.com/ESMValGroup/ESMValCore/pull/2087>`__) `Peter Kalverla <https://github.com/Peter9192>`__
-  Avoid realizing data in preprocessor function :func:`esmvalcore.preprocessor.concatenate` when cubes overlap (`#2109 <https://github.com/ESMValGroup/ESMValCore/pull/2109>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Documentation
~~~~~~~~~~~~~

-  Remove unneeded sphinxcontrib extension (`#2047 <https://github.com/ESMValGroup/ESMValCore/pull/2047>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Show ESMValTool logo on `PyPI webpage <https://pypi.org/project/ESMValCore/>`__ (`#2065 <https://github.com/ESMValGroup/ESMValCore/pull/2065>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Fix gitter badge in README (`#2118 <https://github.com/ESMValGroup/ESMValCore/pull/2118>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Add changelog for v2.9.0 (`#2088 <https://github.com/ESMValGroup/ESMValCore/pull/2088>`__ and `#2123 <https://github.com/ESMValGroup/ESMValCore/pull/2123>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Pass the :obj:`esmvalcore.config.Session` to fixes (`#1988 <https://github.com/ESMValGroup/ESMValCore/pull/1988>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  ICON: Allowed specifying vertical grid information in recipe (`#2067 <https://github.com/ESMValGroup/ESMValCore/pull/2067>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Allow specifying ``raw_units`` for CESM2, EMAC, and ICON CMORizers (`#2043 <https://github.com/ESMValGroup/ESMValCore/pull/2043>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  ICON: allow specifying horizontal grid file in recipe/extra facets (`#2078 <https://github.com/ESMValGroup/ESMValCore/pull/2078>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fix tas/tos CMIP6: FIO, KACE, MIROC, IITM (`#2061 <https://github.com/ESMValGroup/ESMValCore/pull/2061>`__) `Pep Cos <https://github.com/pepcos>`__
-  Add fix for EC-Earth3-Veg tos calendar (`#2100 <https://github.com/ESMValGroup/ESMValCore/pull/2100>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Correct GISS-E2-1-G ``tos`` units (`#2099 <https://github.com/ESMValGroup/ESMValCore/pull/2099>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Installation
~~~~~~~~~~~~

-  Drop support for Python 3.8 (`#2053 <https://github.com/ESMValGroup/ESMValCore/pull/2053>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add python 3.11 to Github Actions package (conda and PyPI) installation tests (`#2083 <https://github.com/ESMValGroup/ESMValCore/pull/2083>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Remove ``with_mypy`` or ``with-mypy`` optional tool for prospector (`#2108 <https://github.com/ESMValGroup/ESMValCore/pull/2108>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

Preprocessor
~~~~~~~~~~~~

-  Added ``period='hourly'`` for :func:`esmvalcore.preprocessor.climate_statistics` and :func:`esmvalcore.preprocessor.anomalies` (`#2068 <https://github.com/ESMValGroup/ESMValCore/pull/2068>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Support IPCC AR6 regions in :func:`esmvalcore.preprocessor.extract_shape` (`#2008 <https://github.com/ESMValGroup/ESMValCore/pull/2008>`__) `Manuel Schlund <https://github.com/schlunma>`__


.. _changelog-v2-8-1:

v2.8.1
------
Highlights
~~~~~~~~~~
This release adds support for Python 3.11 and includes several bugfixes.

This release includes:

Bug fixes
~~~~~~~~~

-  Pin numpy !=1.24.3 (`#2011 <https://github.com/ESMValGroup/ESMValCore/pull/2011>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Fix a bug in recording provenance for the ``mask_multimodel`` preprocessor (`#1984 <https://github.com/ESMValGroup/ESMValCore/pull/1984>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fix ICON hourly data rounding issues (`#2022 <https://github.com/ESMValGroup/ESMValCore/pull/2022>`__) `Julian Bauer <https://github.com/BauerJul>`__
-  Use the default SSL context when using the ``extract_location`` preprocessor (`#2023 <https://github.com/ESMValGroup/ESMValCore/pull/2023>`__) `Emma Hogan <https://github.com/ehogan>`__
-  Make time-related CMOR fixes work with time dimensions `time1`, `time2`, `time3` (`#1971 <https://github.com/ESMValGroup/ESMValCore/pull/1971>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Always create a cache directory for storing ICON grid files (`#2030 <https://github.com/ESMValGroup/ESMValCore/pull/2030>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fixed altitude <--> pressure level conversion for masked arrays in the ``extract_levels`` preprocessor (`#1999 <https://github.com/ESMValGroup/ESMValCore/pull/1999>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Allowed ignoring of scalar time coordinates in the ``multi_model_statistics`` preprocessor (`#1961 <https://github.com/ESMValGroup/ESMValCore/pull/1961>`__) `Manuel Schlund <https://github.com/schlunma>`__

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Add support for hourly ICON data (`#1990 <https://github.com/ESMValGroup/ESMValCore/pull/1990>`__) `Julian Bauer <https://github.com/BauerJul>`__
-  Fix areacello in BCC-CSM2-MR (`#1993 <https://github.com/ESMValGroup/ESMValCore/pull/1993>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__

Installation
~~~~~~~~~~~~

-  Add support for Python=3.11 (`#1832 <https://github.com/ESMValGroup/ESMValCore/pull/1832>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Modernize conda lock file creation workflow with mamba, Mambaforge etc (`#2027 <https://github.com/ESMValGroup/ESMValCore/pull/2027>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Pin `libnetcdf!=4.9.1` (`#2072 <https://github.com/ESMValGroup/ESMValCore/pull/2072>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__

Documentation
~~~~~~~~~~~~~
-  Add changelog for v2.8.1 (`#2079 <https://github.com/ESMValGroup/ESMValCore/pull/2079>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Automatic testing
~~~~~~~~~~~~~~~~~

-  Use mocked `geopy.geocoders.Nominatim` to avoid `ReadTimeoutError` (`#2005 <https://github.com/ESMValGroup/ESMValCore/pull/2005>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Update pre-commit hooks (`#2020 <https://github.com/ESMValGroup/ESMValCore/pull/2020>`__) `Bouwe Andela <https://github.com/bouweandela>`__


.. _changelog-v2-8-0:


v2.8.0
------
Highlights
~~~~~~~~~~

-  ESMValCore now supports wildcards in recipes and offers improved support for
   ancillary variables and dataset versioning thanks to contributions by
   `Bouwe Andela <https://github.com/bouweandela>`__. For details, see
   :ref:`Automatically populating a recipe with all available datasets <dataset_wildcards>`
   and :ref:`Defining supplementary variables <supplementary_variables>`.
-  Support for CORDEX datasets in a rotated pole coordinate system has been
   added by `sloosvel <https://github.com/sloosvel>`__.
-  Native :ref:`ICON <read_icon>` output is now made UGRID-compliant
   on-the-fly to unlock the use of more sophisticated regridding algorithms,
   thanks to `Manuel Schlund <https://github.com/schlunma>`__.
-  The Python API has been extended with the addition of three
   modules: :mod:`esmvalcore.config`, :mod:`esmvalcore.dataset`, and
   :mod:`esmvalcore.local`, all these features courtesy of
   `Bouwe Andela <https://github.com/bouweandela>`__. For details, see our new
   example :doc:`example-notebooks`.
-  The preprocessor :func:`~esmvalcore.preprocessor.multi_model_statistics`
   has been extended to support more use-cases thanks to contributions by
   `Manuel Schlund <https://github.com/schlunma>`__. For details, see
   :ref:`Multi-model statistics <multi-model statistics>`.

This release includes:

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Please read the descriptions of the linked pull requests for detailed upgrade instructions.

-  The algorithm for automatically defining the ancillary variables and cell
   measures has been improved
   (`#1609 <https://github.com/ESMValGroup/ESMValCore/pull/1609>`_)
   `Bouwe Andela <https://github.com/bouweandela>`__.
   If this does not work as expected, more examples of how to adapt your recipes
   are given
   `here <https://github.com/ESMValGroup/ESMValCore/pull/1609#Backward-incompatible-changes>`__
   and in the corresponding sections of the
   :ref:`recipe documentation <supplementary_variables>` and the
   :ref:`preprocessor documentation <preprocessors_using_supplementary_variables>`.
-  Remove deprecated features scheduled for removal in v2.8.0 or earlier
   (`#1826 <https://github.com/ESMValGroup/ESMValCore/pull/1826>`__)
   `Manuel Schlund <https://github.com/schlunma>`__.
   Removed ``esmvalcore.iris_helpers.var_name_constraint`` (has been deprecated
   in v2.6.0; please use :class:`iris.NameConstraint` with the keyword argument
   ``var_name`` instead) and the option ``always_use_ne_mask`` for
   :func:`esmvalcore.preprocessor.mask_landsea` (has been deprecated in v2.5.0;
   the same behavior can now be achieved by specifying ``supplementary_variables``.
-  No files will be found if a non-existent version of a dataset is specified
   (`#1835 <https://github.com/ESMValGroup/ESMValCore/pull/1835>`_)
   `Bouwe Andela <https://github.com/bouweandela>`__. If a ``version`` of a
   dataset is specified in the recipe, the tool will now search for exactly that
   version, instead of simply using the latest version. Therefore, it is
   necessary to make sure that the version number in the directory tree matches
   with the version number in the recipe to find the files.
-  The default filename template for obs4MIPs has been updated to better match
   filenames used in this project in
   (`#1866 <https://github.com/ESMValGroup/ESMValCore/pull/1866>`__)
   `Bouwe Andela <https://github.com/bouweandela>`__. This may cause issues
   if you are storing all the files for obs4MIPs in a directory with no
   subdirectories per dataset.

Deprecations
~~~~~~~~~~~~
Please read the descriptions of the linked pull requests for detailed upgrade instructions.

-  Various configuration related options that are now available through
   :mod:`esmvalcore.config` have been deprecated (`#1769 <https://github.com/ESMValGroup/ESMValCore/pull/1769>`_) `Bouwe Andela <https://github.com/bouweandela>`__.
-  The ``fx_variables`` preprocessor argument and related features have been
   deprecated (`#1609`_)
   `Bouwe Andela <https://github.com/bouweandela>`__.
   See `here <https://github.com/ESMValGroup/ESMValCore/pull/1609#Deprecations>`__
   for more information.
-  Combined ``offline`` and ``always_search_esgf`` into a single option ``search_esgf``
   (`#1935 <https://github.com/ESMValGroup/ESMValCore/pull/1935>`_)
   `Manuel Schlund <https://github.com/schlunma>`__. The configuration
   option/command line argument ``offline`` has been deprecated in favor of
   ``search_esgf``. The previous ``offline: true`` is now ``search_esgf: never``
   (the default); the previous ``offline: false`` is now
   ``search_esgf: when_missing``. More details on how to adapt your workflow
   regarding these new options are given in `#1935`_ and the
   `documentation <https://docs.esmvaltool.org/projects/ESMValCore/en/latest/quickstart/configure.html?highlight=search_esgf#user-configuration-file>`__.
-  :func:`esmvalcore.preprocessor.cleanup` has been deprecated (`#1949 <https://github.com/ESMValGroup/ESMValCore/pull/1949>`_)
   `Manuel Schlund <https://github.com/schlunma>`__. Please do not use this
   anymore in the recipe (it is not necessary).

Python API
~~~~~~~~~~

-  Support searching ESGF for a specific version of a file and add :obj:`esmvalcore.esgf.ESGFFile.facets` (`#1822 <https://github.com/ESMValGroup/ESMValCore/pull/1822>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix issues with searching for files on ESGF (`#1863 <https://github.com/ESMValGroup/ESMValCore/pull/1863>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Move the :mod:`esmvalcore.experimental.config` module to  :mod:`esmvalcore.config` (`#1769`_) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add :mod:`esmvalcore.local`, a module to search data on the local filesystem (`#1835`_) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add :mod:`esmvalcore.dataset` module (`#1877 <https://github.com/ESMValGroup/ESMValCore/pull/1877>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Bug fixes
~~~~~~~~~

-  Import from :mod:`esmvalcore.config` in the :mod:`esmvalcore.experimental` module (`#1816 <https://github.com/ESMValGroup/ESMValCore/pull/1816>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Added scalar coords of input cubes to output of esmpy_regrid (`#1811 <https://github.com/ESMValGroup/ESMValCore/pull/1811>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fix severe bug in :func:`esmvalcore.preprocessor.mask_fillvalues` (`#1823 <https://github.com/ESMValGroup/ESMValCore/pull/1823>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fix LWP of ICON on-the-fly CMORizer (`#1839 <https://github.com/ESMValGroup/ESMValCore/pull/1839>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fixed issue in irregular regridding regarding scalar coordinates (`#1845 <https://github.com/ESMValGroup/ESMValCore/pull/1845>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Update product attributes and `metadata.yml` with cube metadata before saving files (`#1837 <https://github.com/ESMValGroup/ESMValCore/pull/1837>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Remove an extra space character from a filename (`#1883 <https://github.com/ESMValGroup/ESMValCore/pull/1883>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Improve resilience of ESGF search (`#1869 <https://github.com/ESMValGroup/ESMValCore/pull/1869>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix issue with no files found if timerange start/end differs in length (`#1880 <https://github.com/ESMValGroup/ESMValCore/pull/1880>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add `driver` and `sub_experiment` tags to generate dataset aliases (`#1886 <https://github.com/ESMValGroup/ESMValCore/pull/1886>`__) `sloosvel <https://github.com/sloosvel>`__
-  Fixed time points of native CESM2 output (`#1772 <https://github.com/ESMValGroup/ESMValCore/pull/1772>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fix type hints for Python versions < 3.10 (`#1897 <https://github.com/ESMValGroup/ESMValCore/pull/1897>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fixed `set_range_in_0_360` for dask arrays (`#1919 <https://github.com/ESMValGroup/ESMValCore/pull/1919>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Made equalized attributes in concatenated cubes consistent across runs (`#1783 <https://github.com/ESMValGroup/ESMValCore/pull/1783>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fix issue with reading dates from files (`#1936 <https://github.com/ESMValGroup/ESMValCore/pull/1936>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add institute name used on ESGF for CMIP5 CanAM4, CanCM4, and CanESM2 (`#1937 <https://github.com/ESMValGroup/ESMValCore/pull/1937>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix issue where data was not loaded and saved (`#1962 <https://github.com/ESMValGroup/ESMValCore/pull/1962>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix type hints for Python 3.8 (`#1795 <https://github.com/ESMValGroup/ESMValCore/pull/1795>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Update the institute facet of the CSIRO-Mk3L-1-2 model (`#1966 <https://github.com/ESMValGroup/ESMValCore/pull/1966>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Fixed race condition that may result in errors in :func:`esmvalcore.preprocessor.cleanup` (`#1949`_) `Manuel Schlund <https://github.com/schlunma>`__
-  Update notebook so it uses supplementaries instead of ancillaries (`#1945 <https://github.com/ESMValGroup/ESMValCore/pull/1945>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Documentation
~~~~~~~~~~~~~

-  Fix anaconda badge in README (`#1759 <https://github.com/ESMValGroup/ESMValCore/pull/1759>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Fix mistake in the documentation of :obj:`esmvalcore.esgf.find_files` (`#1784 <https://github.com/ESMValGroup/ESMValCore/pull/1784>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Support linking to "stable" ESMValTool version on readthedocs (`#1608 <https://github.com/ESMValGroup/ESMValCore/pull/1608>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Updated ICON doc with information on usage of extract_levels preprocessor (`#1903 <https://github.com/ESMValGroup/ESMValCore/pull/1903>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Add changelog for latest released version v2.7.1 (`#1905 <https://github.com/ESMValGroup/ESMValCore/pull/1905>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Update `preprocessor.rst` due to renaming of NCEP dataset to NCEP-NCAR-R1 (`#1908 <https://github.com/ESMValGroup/ESMValCore/pull/1908>`__) `Birgit Hassler <https://github.com/hb326>`__
-  Replace timerange nested lists in docs with overview table (`#1940 <https://github.com/ESMValGroup/ESMValCore/pull/1940>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Updated section "backward compatibility" in `contributing.rst` (`#1918 <https://github.com/ESMValGroup/ESMValCore/pull/1918>`__) `Axel Lauer <https://github.com/axel-lauer>`__
-  Add link to ESMValTool release procedure steps (`#1957 <https://github.com/ESMValGroup/ESMValCore/pull/1957>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Synchronize documentation table of contents with ESMValTool (`#1958 <https://github.com/ESMValGroup/ESMValCore/pull/1958>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Improvements
~~~~~~~~~~~~

-  Support wildcards in the recipe and improve support for ancillary variables and dataset versioning (`#1609`_) `Bouwe Andela <https://github.com/bouweandela>`__. More details on how to adapt your recipes are given in the corresponding pull request description and in the corresponding sections of the `recipe documentation <https://docs.esmvaltool.org/projects/esmvalcore/en/latest/recipe/overview.html#defining-supplementary-variables-ancillary-variables-and-cell-measures>`__ and the `preprocessor documentation <https://docs.esmvaltool.org/projects/esmvalcore/en/latest/recipe/preprocessor.html#preprocessors-using-supplementary-variables>`__.
-  Create a session directory with suffix "-1", "-2", etc if it already exists (`#1818 <https://github.com/ESMValGroup/ESMValCore/pull/1818>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Message for users when they use esmvaltool executable from esmvalcore only (`#1831 <https://github.com/ESMValGroup/ESMValCore/pull/1831>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Order recipe output in index.html (`#1899 <https://github.com/ESMValGroup/ESMValCore/pull/1899>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Improve reading facets from ESGF search results (`#1920 <https://github.com/ESMValGroup/ESMValCore/pull/1920>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Fix rotated coordinate grids and `tas` and `pr` for CORDEX datasets (`#1765 <https://github.com/ESMValGroup/ESMValCore/pull/1765>`__) `sloosvel <https://github.com/sloosvel>`__
-  Made ICON output UGRID-compliant (on-the-fly) (`#1664 <https://github.com/ESMValGroup/ESMValCore/pull/1664>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fix automatic download of ICON grid file and make ICON UGRIDization optional (`default: true`) (`#1922 <https://github.com/ESMValGroup/ESMValCore/pull/1922>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Add siconc fixes for EC-Earth3-Veg and EC-Earth3-Veg-LR models (`#1771 <https://github.com/ESMValGroup/ESMValCore/pull/1771>`__) `Evgenia Galytska <https://github.com/egalytska>`__
-  Fix siconc in KIOST-ESM (`#1829 <https://github.com/ESMValGroup/ESMValCore/pull/1829>`__) `Lisa Bock <https://github.com/LisaBock>`__
-  Extension of ERA5 CMORizer (variable cl) (`#1850 <https://github.com/ESMValGroup/ESMValCore/pull/1850>`__) `Axel Lauer <https://github.com/axel-lauer>`__
-  Add standard variable names for EMAC (`#1853 <https://github.com/ESMValGroup/ESMValCore/pull/1853>`__) `FranziskaWinterstein <https://github.com/FranziskaWinterstein>`__
-  Fix for FGOALS-f3-L clt (`#1928 <https://github.com/ESMValGroup/ESMValCore/pull/1928>`__) `Lisa Bock <https://github.com/LisaBock>`__

Installation
~~~~~~~~~~~~

-  Add all deps to the conda-forge environment and suppress installing and reinstalling deps with pip at readthedocs builds (`#1786 <https://github.com/ESMValGroup/ESMValCore/pull/1786>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Pin netCDF4<1.6.1 (`#1805 <https://github.com/ESMValGroup/ESMValCore/pull/1805>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Unpin NetCF4 (`#1814 <https://github.com/ESMValGroup/ESMValCore/pull/1814>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Unpin flake8 (`#1820 <https://github.com/ESMValGroup/ESMValCore/pull/1820>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Add iris-esmf-regrid as a dependency (`#1809 <https://github.com/ESMValGroup/ESMValCore/pull/1809>`__) `sloosvel <https://github.com/sloosvel>`__
-  Pin esmpy<8.4 (`#1871 <https://github.com/ESMValGroup/ESMValCore/pull/1871>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Update esmpy import for ESMF v8.4.0 (`#1876 <https://github.com/ESMValGroup/ESMValCore/pull/1876>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Preprocessor
~~~~~~~~~~~~
-  Allow :func:`esmvalcore.preprocessor.multi_model_statistics` on cubes with arbitrary dimensions  (`#1808 <https://github.com/ESMValGroup/ESMValCore/pull/1808>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Smarter removal of coordinate metadata in :func:`esmvalcore.preprocessor.multi_model_statistics` preprocessor (`#1813 <https://github.com/ESMValGroup/ESMValCore/pull/1813>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Allowed usage of :func:`esmvalcore.preprocessor.multi_model_statistics` on single cubes/products (`#1849 <https://github.com/ESMValGroup/ESMValCore/pull/1849>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Allowed usage of :func:`esmvalcore.preprocessor.multi_model_statistics` on cubes with identical ``name()`` and ``units`` (but e.g. different long_name) (`#1921 <https://github.com/ESMValGroup/ESMValCore/pull/1921>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Allowed ignoring scalar coordinates in :func:`esmvalcore.preprocessor.multi_model_statistics` (`#1934 <https://github.com/ESMValGroup/ESMValCore/pull/1934>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Refactored :func:`esmvalcore.preprocessor.regrid` and removed unnecessary code not needed anymore due to new iris version (`#1898 <https://github.com/ESMValGroup/ESMValCore/pull/1898>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Do not realise coordinates during CMOR check (`#1912 <https://github.com/ESMValGroup/ESMValCore/pull/1912>`__) `sloosvel <https://github.com/sloosvel>`__
-  Make :func:`esmvalcore.preprocessor.extract_volume` work with closed and mixed intervals and allow nearest value selection (`#1930 <https://github.com/ESMValGroup/ESMValCore/pull/1930>`__) `sloosvel <https://github.com/sloosvel>`__

Release
~~~~~~~
-  Changelog for `v2.8.0rc1` (`#1952 <https://github.com/ESMValGroup/ESMValCore/pull/1952>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Increase version number for ESMValCore `v2.8.0rc1` (`#1955 <https://github.com/ESMValGroup/ESMValCore/pull/1955>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Changelog for `v2.8.0rc2` (`#1959 <https://github.com/ESMValGroup/ESMValCore/pull/1959>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Increase version number for ESMValCore `v2.8.0rc2` (`#1973 <https://github.com/ESMValGroup/ESMValCore/pull/1973>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Changelog for `v2.8.0` (`#1978 <https://github.com/ESMValGroup/ESMValCore/pull/1978>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Increase version number for ESMValCore `v2.8.0` (`#1983 <https://github.com/ESMValGroup/ESMValCore/pull/1983>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__

Automatic testing
~~~~~~~~~~~~~~~~~

-  Set implicit optional to true in `mypy` config to avert side effects and test fails from new mypy version (`#1790 <https://github.com/ESMValGroup/ESMValCore/pull/1790>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Remove duplicate `implicit_optional = True` line in ``setup.cfg`` (`#1791 <https://github.com/ESMValGroup/ESMValCore/pull/1791>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Fix failing test due to missing sample data (`#1797 <https://github.com/ESMValGroup/ESMValCore/pull/1797>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Remove outdated cmor_table facet from data finder tests (`#1798 <https://github.com/ESMValGroup/ESMValCore/pull/1798>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Modernize tests for :func:`esmvalcore.preprocessor.save` (`#1799 <https://github.com/ESMValGroup/ESMValCore/pull/1799>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  No more sequential tests since SegFaults were not noticed anymore (`#1819 <https://github.com/ESMValGroup/ESMValCore/pull/1819>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Update pre-commit configuration (`#1821 <https://github.com/ESMValGroup/ESMValCore/pull/1821>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Updated URL of ICON grid file used for testing (`#1914 <https://github.com/ESMValGroup/ESMValCore/pull/1914>`__) `Manuel Schlund <https://github.com/schlunma>`__

Variable Derivation
~~~~~~~~~~~~~~~~~~~

-  Add derivation of sea ice extent (`#1695 <https://github.com/ESMValGroup/ESMValCore/pull/1695>`__) `sloosvel <https://github.com/sloosvel>`__


.. _changelog-v2-7-1:


v2.7.1
------
Highlights
~~~~~~~~~~

This is a bugfix release where we unpin `cf-units` to allow the latest `iris=3.4.0` to be installed. It also includes an update to the default configuration used when searching the ESGF for files, to account for a recent change of the CEDA ESGF index node hostname. The changelog contains only changes that were made to the ``main`` branch.

Installation
~~~~~~~~~~~~

- Set the version number on the development branches to one minor version more than the previous release (`#1854 <https://github.com/ESMValGroup/ESMValCore/pull/1854>`__) `Bouwe Andela <https://github.com/bouweandela>`__
- Unpin cf-units (`#1770 <https://github.com/ESMValGroup/ESMValCore/pull/1770>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Bug fixes
~~~~~~~~~

- Improve error handling if an esgf index node is offline (`#1834 <https://github.com/ESMValGroup/ESMValCore/pull/1834>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Automatic testing
~~~~~~~~~~~~~~~~~

- Removed unnecessary test that fails with iris 3.4.0 (`#1846 <https://github.com/ESMValGroup/ESMValCore/pull/1846>`__) `Manuel Schlund <https://github.com/schlunma>`__
- Update CEDA ESGF index node hostname (`#1838 <https://github.com/ESMValGroup/ESMValCore/pull/1838>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__


.. _changelog-v2-7-0:


v2.7.0
------
Highlights
~~~~~~~~~~

-  We have a new preprocessor function called `'rolling_window_statistics' <https://docs.esmvaltool.org/projects/ESMValCore/en/latest/recipe/preprocessor.html#rolling-window-statistics>`__ implemented by `Liza Malinina <https://github.com/malininae>`__
-  We have improved the support for native models, refactored native model fixes by adding common base class `NativeDatasetFix`, changed default DRS for reading native ICON output, and added tests for input/output filenames for `ICON <https://docs.esmvaltool.org/projects/ESMValCore/en/latest/quickstart/find_data.html#icon>`__ and `EMAC <https://docs.esmvaltool.org/projects/ESMValCore/en/latest/quickstart/find_data.html#emac>`__ on-the-fly CMORizer, all these features courtesy of `Manuel Schlund <https://github.com/schlunma>`__
-  Performance of preprocessor functions that use time dimensions has been sped up by **two orders of magnitude** thanks to contributions by `Bouwe Andela <https://github.com/bouweandela>`__

This release includes:

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Change default DRS for reading native ICON output (`#1705 <https://github.com/ESMValGroup/ESMValCore/pull/1705>`__) `Manuel Schlund <https://github.com/schlunma>`__

Bug fixes
~~~~~~~~~

-  Add support for regions stored as MultiPolygon to extract_shape preprocessor (`#1670 <https://github.com/ESMValGroup/ESMValCore/pull/1670>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fixed type annotations for Python 3.8 (`#1700 <https://github.com/ESMValGroup/ESMValCore/pull/1700>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Core `_io.concatenate()` may fail due to case when one of the cubes is scalar - this fixes that (`#1715 <https://github.com/ESMValGroup/ESMValCore/pull/1715>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Pick up esmvalcore badge instead of esmvaltool one in README (`#1749 <https://github.com/ESMValGroup/ESMValCore/pull/1749>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Restore support for scalar cubes to time selection preprocessor functions (`#1750 <https://github.com/ESMValGroup/ESMValCore/pull/1750>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix calculation of precipitation flux in EMAC on-the-fly CMORizer (`#1755 <https://github.com/ESMValGroup/ESMValCore/pull/1755>`__) `Manuel Schlund <https://github.com/schlunma>`__

Deprecations
~~~~~~~~~~~~

-  Remove deprecation warning for regrid schemes already deprecated for v2.7.0 (`#1753 <https://github.com/ESMValGroup/ESMValCore/pull/1753>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

Documentation
~~~~~~~~~~~~~

-  Add Met Office Installation Method (`#1692 <https://github.com/ESMValGroup/ESMValCore/pull/1692>`__) `mo-tgeddes <https://github.com/mo-tgeddes>`__
-  Add MO-paths to config file (`#1709 <https://github.com/ESMValGroup/ESMValCore/pull/1709>`__) `mo-tgeddes <https://github.com/mo-tgeddes>`__
-  Update MO obs4MIPs paths in the user configuration file (`#1734 <https://github.com/ESMValGroup/ESMValCore/pull/1734>`__) `mo-tgeddes <https://github.com/mo-tgeddes>`__
-  Update `Making a release` section of the documentation (`#1689 <https://github.com/ESMValGroup/ESMValCore/pull/1689>`__) `sloosvel <https://github.com/sloosvel>`__
-  Added changelog for v2.7.0 (`#1746 <https://github.com/ESMValGroup/ESMValCore/pull/1746>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  update CITATION.cff file with 2.7.0 release info (`#1757 <https://github.com/ESMValGroup/ESMValCore/pull/1757>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

Improvements
~~~~~~~~~~~~

-  New preprocessor function 'rolling_window_statistics' (`#1702 <https://github.com/ESMValGroup/ESMValCore/pull/1702>`__) `Liza Malinina <https://github.com/malininae>`__
-  Remove `pytest_flake8` plugin and use `flake8` instead (`#1722 <https://github.com/ESMValGroup/ESMValCore/pull/1722>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Added CESM2 CMORizer (`#1678 <https://github.com/ESMValGroup/ESMValCore/pull/1678>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Speed up functions that use time dimension (`#1713 <https://github.com/ESMValGroup/ESMValCore/pull/1713>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Modernize and minimize pylint configuration (`#1726 <https://github.com/ESMValGroup/ESMValCore/pull/1726>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Refactored native model fixes by adding common base class `NativeDatasetFix` (`#1694 <https://github.com/ESMValGroup/ESMValCore/pull/1694>`__) `Manuel Schlund <https://github.com/schlunma>`__

Installation
~~~~~~~~~~~~

-  Pin `netCDF4 != 1.6.1` since that seems to throw a flurry of Segmentation Faults (`#1724 <https://github.com/ESMValGroup/ESMValCore/pull/1724>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

Automatic testing
~~~~~~~~~~~~~~~~~

-  Pin `flake8<5.0.0` since Circle CI tests are failing copiously (`#1698 <https://github.com/ESMValGroup/ESMValCore/pull/1698>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Added tests for input/output filenames for ICON and EMAC on-the-fly CMORizer (`#1718 <https://github.com/ESMValGroup/ESMValCore/pull/1718>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fix failed tests for Python<3.10 resulting from typing (`#1748 <https://github.com/ESMValGroup/ESMValCore/pull/1748>`__) `Manuel Schlund <https://github.com/schlunma>`__

.. _changelog-v2-6-0:

v2.6.0
------

Highlights
~~~~~~~~~~

- A new set of CMOR fixes is now available in order to load native EMAC model output and CMORize it on the fly. For details, see :ref:`Supported native models: EMAC <read_emac>`.
- The version number of ESMValCore is now automatically generated using `setuptools_scm <https://github.com/pypa/setuptools_scm/#default-versioning-scheme>`__, which extracts Python package versions from git metadata.

This release includes

Deprecations
~~~~~~~~~~~~

-  Deprecate the function `esmvalcore.var_name_constraint` (`#1592 <https://github.com/ESMValGroup/ESMValCore/pull/1592>`__) `Manuel Schlund <https://github.com/schlunma>`__. This function is scheduled for removal in v2.8.0. Please use :class:`iris.NameConstraint` with the keyword argument `var_name` instead: this is an exact replacement.

Bug fixes
~~~~~~~~~

-  Added `start_year` and `end_year` attributes to derived variables (`#1547 <https://github.com/ESMValGroup/ESMValCore/pull/1547>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Show all results on recipe results webpage (`#1560 <https://github.com/ESMValGroup/ESMValCore/pull/1560>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Regridding regular grids with similar coordinates  (`#1567 <https://github.com/ESMValGroup/ESMValCore/pull/1567>`__) `Tomas Lovato <https://github.com/tomaslovato>`__
-  Fix timerange wildcard search when deriving variables or downloading files (`#1562 <https://github.com/ESMValGroup/ESMValCore/pull/1562>`__) `sloosvel <https://github.com/sloosvel>`__
-  Fix `force_derivation` bug (`#1627 <https://github.com/ESMValGroup/ESMValCore/pull/1627>`__) `sloosvel <https://github.com/sloosvel>`__
-  Correct `build-and-deploy-on-pypi` action (`#1634 <https://github.com/ESMValGroup/ESMValCore/pull/1634>`__) `sloosvel <https://github.com/sloosvel>`__
-  Apply `clip_timerange` to time dependent fx variables (`#1603 <https://github.com/ESMValGroup/ESMValCore/pull/1603>`__) `sloosvel <https://github.com/sloosvel>`__
-  Correctly handle requests.exceptions.ConnectTimeout when an ESGF index node is offline (`#1638 <https://github.com/ESMValGroup/ESMValCore/pull/1638>`__) `Bouwe Andela <https://github.com/bouweandela>`__

CMOR standard
~~~~~~~~~~~~~

-  Added custom CMOR tables used for EMAC CMORizer (`#1599 <https://github.com/ESMValGroup/ESMValCore/pull/1599>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Extended ICON CMORizer (`#1549 <https://github.com/ESMValGroup/ESMValCore/pull/1549>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Add CMOR check exception for a basin coord named sector (`#1612 <https://github.com/ESMValGroup/ESMValCore/pull/1612>`__) `David Hohn <https://github.com/dhohn>`__
-  Custom user-defined location for custom CMOR tables (`#1625 <https://github.com/ESMValGroup/ESMValCore/pull/1625>`__) `Manuel Schlund <https://github.com/schlunma>`__

Containerization
~~~~~~~~~~~~~~~~

-  Remove update command in Dockerfile (`#1630 <https://github.com/ESMValGroup/ESMValCore/pull/1630>`__) `sloosvel <https://github.com/sloosvel>`__

Community
~~~~~~~~~

-  Add David Hohn to contributors' list (`#1586 <https://github.com/ESMValGroup/ESMValCore/pull/1586>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

Documentation
~~~~~~~~~~~~~

-  [Github Actions Docs] Full explanation on how to use the GA test triggered by PR comment and added docs link for GA hosted runners  (`#1553 <https://github.com/ESMValGroup/ESMValCore/pull/1553>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Update the command for building the documentation (`#1556 <https://github.com/ESMValGroup/ESMValCore/pull/1556>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Update documentation on running the tool (`#1400 <https://github.com/ESMValGroup/ESMValCore/pull/1400>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add support for DKRZ-Levante (`#1558 <https://github.com/ESMValGroup/ESMValCore/pull/1558>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Improved documentation on native dataset support (`#1559 <https://github.com/ESMValGroup/ESMValCore/pull/1559>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Tweak `extract_point` preprocessor: explain what it returns if one point coord outside cube and add explicit test  (`#1584 <https://github.com/ESMValGroup/ESMValCore/pull/1584>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Update CircleCI, readthedocs, and Docker configuration (`#1588 <https://github.com/ESMValGroup/ESMValCore/pull/1588>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Remove support for Mistral in `config-user.yml` (`#1620 <https://github.com/ESMValGroup/ESMValCore/pull/1620>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Add changelog for v2.6.0rc1 (`#1633 <https://github.com/ESMValGroup/ESMValCore/pull/1633>`__) `sloosvel <https://github.com/sloosvel>`__
-  Add a note on transferring permissions to the release manager (`#1645 <https://github.com/ESMValGroup/ESMValCore/pull/1645>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add documentation on building and uploading Docker images (`#1644 <https://github.com/ESMValGroup/ESMValCore/pull/1644>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Update documentation on ESMValTool module at DKRZ (`#1647 <https://github.com/ESMValGroup/ESMValCore/pull/1647>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Expanded information on deprecations in changelog (`#1658 <https://github.com/ESMValGroup/ESMValCore/pull/1658>`__) `Manuel Schlund <https://github.com/schlunma>`__

Improvements
~~~~~~~~~~~~

-  Removed trailing whitespace in custom CMOR tables (`#1564 <https://github.com/ESMValGroup/ESMValCore/pull/1564>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Try searching multiple ESGF index nodes (`#1561 <https://github.com/ESMValGroup/ESMValCore/pull/1561>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add CMIP6 `amoc` derivation case and add a test (`#1577 <https://github.com/ESMValGroup/ESMValCore/pull/1577>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Added EMAC CMORizer (`#1554 <https://github.com/ESMValGroup/ESMValCore/pull/1554>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Improve performance of `volume_statistics` (`#1545 <https://github.com/ESMValGroup/ESMValCore/pull/1545>`__) `sloosvel <https://github.com/sloosvel>`__

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Fixes of ocean variables in multiple CMIP6 datasets (`#1566 <https://github.com/ESMValGroup/ESMValCore/pull/1566>`__) `Tomas Lovato <https://github.com/tomaslovato>`__
-  Ensure lat/lon bounds in FGOALS-l3 atmos variables are contiguous (`#1571 <https://github.com/ESMValGroup/ESMValCore/pull/1571>`__) `sloosvel <https://github.com/sloosvel>`__
-  Added `AllVars` fix for CMIP6's ICON-ESM-LR (`#1582 <https://github.com/ESMValGroup/ESMValCore/pull/1582>`__) `Manuel Schlund <https://github.com/schlunma>`__

Installation
~~~~~~~~~~~~

-  Removed `package/meta.yml` (`#1540 <https://github.com/ESMValGroup/ESMValCore/pull/1540>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Pinned iris>=3.2.1 (`#1552 <https://github.com/ESMValGroup/ESMValCore/pull/1552>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Use setuptools-scm to automatically generate the version number (`#1578 <https://github.com/ESMValGroup/ESMValCore/pull/1578>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Pin cf-units to lower than 3.1.0 to temporarily avoid changes within new version related to calendars (`#1659 <https://github.com/ESMValGroup/ESMValCore/pull/1659>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

Preprocessor
~~~~~~~~~~~~

-  Allowed special case for unit conversion of precipitation (`kg m-2 s-1` <--> `mm day-1`) (`#1574 <https://github.com/ESMValGroup/ESMValCore/pull/1574>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Add general `extract_coordinate_points` preprocessor (`#1581 <https://github.com/ESMValGroup/ESMValCore/pull/1581>`__) `sloosvel <https://github.com/sloosvel>`__
-  Add preprocessor `accumulate_coordinate` (`#1281 <https://github.com/ESMValGroup/ESMValCore/pull/1281>`__) `Javier Vegas-Regidor <https://github.com/jvegreg>`__
-  Add `axis_statistics` and improve `depth_integration` (`#1589 <https://github.com/ESMValGroup/ESMValCore/pull/1589>`__) `sloosvel <https://github.com/sloosvel>`__

Release
~~~~~~~

-  Increase version number for ESMValCore v2.6.0rc1 (`#1632 <https://github.com/ESMValGroup/ESMValCore/pull/1632>`__) `sloosvel <https://github.com/sloosvel>`__
-  Update changelog and version for 2.6rc3 (`#1646 <https://github.com/ESMValGroup/ESMValCore/pull/1646>`__) `sloosvel <https://github.com/sloosvel>`__
-  Add changelog for rc4 (`#1662 <https://github.com/ESMValGroup/ESMValCore/pull/1662>`__) `sloosvel <https://github.com/sloosvel>`__


Automatic testing
~~~~~~~~~~~~~~~~~

-  Refresh CircleCI cache weekly (`#1597 <https://github.com/ESMValGroup/ESMValCore/pull/1597>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Use correct cache restore key on CircleCI (`#1598 <https://github.com/ESMValGroup/ESMValCore/pull/1598>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Install git and ssh before checking out code on CircleCI (`#1601 <https://github.com/ESMValGroup/ESMValCore/pull/1601>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fetch all history in Github Action tests (`#1622 <https://github.com/ESMValGroup/ESMValCore/pull/1622>`__) `sloosvel <https://github.com/sloosvel>`__
-  Test Github Actions dashboard badge from meercode.io (`#1640 <https://github.com/ESMValGroup/ESMValCore/pull/1640>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Improve esmvalcore.esgf unit test (`#1650 <https://github.com/ESMValGroup/ESMValCore/pull/1650>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Variable Derivation
~~~~~~~~~~~~~~~~~~~

-  Added derivation of `hfns` (`#1594 <https://github.com/ESMValGroup/ESMValCore/pull/1594>`__) `Manuel Schlund <https://github.com/schlunma>`__

.. _changelog-v2-5-0:

v2.5.0
------

Highlights
~~~~~~~~~~

-  The new preprocessor :func:`~esmvalcore.preprocessor.extract_location` can extract arbitrary locations on the Earth using the `geopy <https://pypi.org/project/geopy/>`__ package that connects to OpenStreetMap. For details, see :ref:`Extract location <extract_location>`.
-  Time ranges can now be extracted using the `ISO 8601 format <https://en.wikipedia.org/wiki/ISO_8601>`_. In addition, wildcards are allowed, which makes the time selection much more flexible. For details, see :ref:`Recipe section: Datasets <Datasets>`.
-  The new preprocessor :func:`~esmvalcore.preprocessor.ensemble_statistics` can calculate arbitrary statistics over all ensemble members of a simulation. In addition, the preprocessor :func:`~esmvalcore.preprocessor.multi_model_statistics` now accepts the keyword ``groupy``, which allows the calculation of multi-model statistics over arbitrary multi-model ensembles. For details, see :ref:`Ensemble statistics <ensemble statistics>` and :ref:`Multi-model statistics <multi-model statistics>`.

This release includes

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Update Cordex section in  `config-developer.yml` (`#1303 <https://github.com/ESMValGroup/ESMValCore/pull/1303>`__) `francesco-cmcc <https://github.com/francesco-cmcc>`__. This changes the naming convention of ESMValCore's output files from CORDEX dataset. This only affects recipes that use CORDEX data. Most likely, no changes in diagnostics are necessary; however, if code relies on the specific naming convention of files, it might need to be adapted.
-  Dropped Python 3.7 (`#1530 <https://github.com/ESMValGroup/ESMValCore/pull/1530>`__) `Manuel Schlund <https://github.com/schlunma>`__. ESMValCore v2.5.0 dropped support for Python 3.7. From now on Python >=3.8 is required to install ESMValCore. The main reason for this is that conda-forge dropped support for Python 3.7 for OSX and arm64 (more details are given `here <https://github.com/ESMValGroup/ESMValTool/issues/2584#issuecomment-1063853630>`__).

Bug fixes
~~~~~~~~~

-  Fix `extract_shape` when fx vars are present (`#1403 <https://github.com/ESMValGroup/ESMValCore/pull/1403>`__) `sloosvel <https://github.com/sloosvel>`__
-  Added support of `extra_facets` to fx variables added by the preprocessor (`#1399 <https://github.com/ESMValGroup/ESMValCore/pull/1399>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Augmented input for derived variables with extra_facets (`#1412 <https://github.com/ESMValGroup/ESMValCore/pull/1412>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Correctly use masked arrays after `unstructured_nearest` regridding (`#1414 <https://github.com/ESMValGroup/ESMValCore/pull/1414>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fixing the broken derivation script for XCH4 (and XCO2) (`#1428 <https://github.com/ESMValGroup/ESMValCore/pull/1428>`__) `Birgit Hassler <https://github.com/hb326>`__
-  Ignore `.pymon-journal` file in test discovery (`#1436 <https://github.com/ESMValGroup/ESMValCore/pull/1436>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Fixed bug that caused automatic download to fail in rare cases (`#1442 <https://github.com/ESMValGroup/ESMValCore/pull/1442>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Add new `JULIA_LOAD_PATH` to diagnostic task test (`#1444 <https://github.com/ESMValGroup/ESMValCore/pull/1444>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Fix provenance file permissions (`#1468 <https://github.com/ESMValGroup/ESMValCore/pull/1468>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fixed usage of `statistics=std_dev` option in multi-model statistics preprocessors (`#1478 <https://github.com/ESMValGroup/ESMValCore/pull/1478>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Removed scalar coordinates `p0` and `ptop` prior to merge in `multi_model_statistics` (`#1471 <https://github.com/ESMValGroup/ESMValCore/pull/1471>`__) `Axel Lauer <https://github.com/axel-lauer>`__
-  Added `dataset` and `alias` attributes to `multi_model_statistics` output (`#1483 <https://github.com/ESMValGroup/ESMValCore/pull/1483>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fixed issues with multi-model-statistics timeranges (`#1486 <https://github.com/ESMValGroup/ESMValCore/pull/1486>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fixed output messages for CMOR logging (`#1494 <https://github.com/ESMValGroup/ESMValCore/pull/1494>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fixed `clip_timerange` if only a single time point is extracted (`#1497 <https://github.com/ESMValGroup/ESMValCore/pull/1497>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fixed chunking in `multi_model_statistics` (`#1500 <https://github.com/ESMValGroup/ESMValCore/pull/1500>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fixed renaming of auxiliary coordinates in `multi_model_statistics` if coordinates are equal (`#1502 <https://github.com/ESMValGroup/ESMValCore/pull/1502>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fixed timerange selection for automatic downloads (`#1517 <https://github.com/ESMValGroup/ESMValCore/pull/1517>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fixed chunking in `multi_model_statistics` (`#1524 <https://github.com/ESMValGroup/ESMValCore/pull/1524>`__) `Manuel Schlund <https://github.com/schlunma>`__

Deprecations
~~~~~~~~~~~~

-  Renamed vertical regridding schemes (`#1429 <https://github.com/ESMValGroup/ESMValCore/pull/1429>`__) `Manuel Schlund <https://github.com/schlunma>`__. Old regridding schemes are supported until v2.7.0. For details, see :ref:`Vertical interpolation schemes <Vertical interpolation schemes>`.

Documentation
~~~~~~~~~~~~~

-  Remove duplicate entries in changelog (`#1391 <https://github.com/ESMValGroup/ESMValCore/pull/1391>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Documentation on how to use HPC central installations (`#1409 <https://github.com/ESMValGroup/ESMValCore/pull/1409>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Correct brackets in preprocessor documentation for list of seasons (`#1420 <https://github.com/ESMValGroup/ESMValCore/pull/1420>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add Python=3.10 to package info, update Circle CI auto install and documentation for Python=3.10 (`#1432 <https://github.com/ESMValGroup/ESMValCore/pull/1432>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Reverted unintentional change in `.zenodo.json` (`#1452 <https://github.com/ESMValGroup/ESMValCore/pull/1452>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Synchronized config-user.yml with version from ESMValTool (`#1453 <https://github.com/ESMValGroup/ESMValCore/pull/1453>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Solved issues in configuration files (`#1457 <https://github.com/ESMValGroup/ESMValCore/pull/1457>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Add direct link to download conda lock file in the install documentation (`#1462 <https://github.com/ESMValGroup/ESMValCore/pull/1462>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  CITATION.cff fix and automatic validation of citation metadata (`#1467 <https://github.com/ESMValGroup/ESMValCore/pull/1467>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Updated documentation on how to deprecate features (`#1426 <https://github.com/ESMValGroup/ESMValCore/pull/1426>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Added reference hook to conda lock in documentation install section (`#1473 <https://github.com/ESMValGroup/ESMValCore/pull/1473>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Increased ESMValCore version to 2.5.0rc1 (`#1477 <https://github.com/ESMValGroup/ESMValCore/pull/1477>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Added changelog for v2.5.0 release (`#1476 <https://github.com/ESMValGroup/ESMValCore/pull/1476>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Increased ESMValCore version to 2.5.0rc2 (`#1487 <https://github.com/ESMValGroup/ESMValCore/pull/1487>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Added some authors to citation and zenodo files (`#1488 <https://github.com/ESMValGroup/ESMValCore/pull/1488>`__) `SarahAlidoost <https://github.com/SarahAlidoost>`__
-  Restored `scipy` intersphinx mapping (`#1491 <https://github.com/ESMValGroup/ESMValCore/pull/1491>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Increased ESMValCore version to 2.5.0rc3 (`#1504 <https://github.com/ESMValGroup/ESMValCore/pull/1504>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fix download instructions for the MSWEP dataset (`#1506 <https://github.com/ESMValGroup/ESMValCore/pull/1506>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Documentation updated for the new cmorizer framework (`#1417 <https://github.com/ESMValGroup/ESMValCore/pull/1417>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Added tests for duplicates in changelog and removed duplicates (`#1508 <https://github.com/ESMValGroup/ESMValCore/pull/1508>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Increased ESMValCore version to 2.5.0rc4 (`#1519 <https://github.com/ESMValGroup/ESMValCore/pull/1519>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Add Github Actions Test badge in README (`#1526 <https://github.com/ESMValGroup/ESMValCore/pull/1526>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Increased ESMValCore version to 2.5.0rc5 (`#1529 <https://github.com/ESMValGroup/ESMValCore/pull/1529>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Increased ESMValCore version to 2.5.0rc6 (`#1532 <https://github.com/ESMValGroup/ESMValCore/pull/1532>`__) `Manuel Schlund <https://github.com/schlunma>`__

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Added fix for AIRS v2.1 (obs4mips) (`#1472 <https://github.com/ESMValGroup/ESMValCore/pull/1472>`__) `Axel Lauer <https://github.com/axel-lauer>`__

Preprocessor
~~~~~~~~~~~~

-  Added bias preprocessor (`#1406 <https://github.com/ESMValGroup/ESMValCore/pull/1406>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Improve error messages when a preprocessor is failing (`#1408 <https://github.com/ESMValGroup/ESMValCore/pull/1408>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Added option to explicitly not use fx variables in preprocessors (`#1416 <https://github.com/ESMValGroup/ESMValCore/pull/1416>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Add `extract_location` preprocessor to extract town, city, mountains etc - anything specifiable by a location (`#1251 <https://github.com/ESMValGroup/ESMValCore/pull/1251>`__) `Javier Vegas-Regidor <https://github.com/jvegreg>`__
-  Add ensemble statistics preprocessor and 'groupby' option for multimodel (`#673 <https://github.com/ESMValGroup/ESMValCore/pull/673>`__) `sloosvel <https://github.com/sloosvel>`__
-  Generic regridding preprocessor (`#1448 <https://github.com/ESMValGroup/ESMValCore/pull/1448>`__) `Klaus Zimmermann <https://github.com/zklaus>`__

Automatic testing
~~~~~~~~~~~~~~~~~

-  Add `pandas` as dependency :panda_face:  (`#1402 <https://github.com/ESMValGroup/ESMValCore/pull/1402>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Fixed tests for python 3.7 (`#1410 <https://github.com/ESMValGroup/ESMValCore/pull/1410>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Remove accessing `.xml()` cube method from test (`#1419 <https://github.com/ESMValGroup/ESMValCore/pull/1419>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Remove flag to use pip 2020 solver from Github Action pip install command on OSX (`#1357 <https://github.com/ESMValGroup/ESMValCore/pull/1357>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Add Python=3.10 to Github Actions and switch to Python=3.10 for the Github Action that builds the PyPi package (`#1430 <https://github.com/ESMValGroup/ESMValCore/pull/1430>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Pin `flake8<4` to keep getting relevant error traces when tests fail with FLAKE8 issues (`#1434 <https://github.com/ESMValGroup/ESMValCore/pull/1434>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Implementing conda lock (`#1164 <https://github.com/ESMValGroup/ESMValCore/pull/1164>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Relocate `pytest-monitor` outputted database `.pymon` so `.pymon-journal` file should not be looked for by `pytest` (`#1441 <https://github.com/ESMValGroup/ESMValCore/pull/1441>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Switch to Mambaforge in Github Actions tests (`#1438 <https://github.com/ESMValGroup/ESMValCore/pull/1438>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Turn off conda lock file creation on any push on `main` branch from Github Action test (`#1489 <https://github.com/ESMValGroup/ESMValCore/pull/1489>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Add DRS path test for IPSLCM files (`#1490 <https://github.com/ESMValGroup/ESMValCore/pull/1490>`__) `Stéphane Sénési <https://github.com/senesis>`__
-  Add a test module that runs tests of `iris` I/O every time we notice serious bugs there (`#1510 <https://github.com/ESMValGroup/ESMValCore/pull/1510>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  [Github Actions] Trigger Github Actions tests (`run-tests.yml` workflow) from a comment in a PR (`#1520 <https://github.com/ESMValGroup/ESMValCore/pull/1520>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Update Linux condalock file (various pull requests) github-actions[bot]

Installation
~~~~~~~~~~~~

-  Move `nested-lookup` dependency to `environment.yml` to be installed from conda-forge instead of PyPi (`#1481 <https://github.com/ESMValGroup/ESMValCore/pull/1481>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Pinned `iris` (`#1511 <https://github.com/ESMValGroup/ESMValCore/pull/1511>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Updated dependencies (`#1521 <https://github.com/ESMValGroup/ESMValCore/pull/1521>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Pinned iris<3.2.0 (`#1525 <https://github.com/ESMValGroup/ESMValCore/pull/1525>`__) `Manuel Schlund <https://github.com/schlunma>`__

Improvements
~~~~~~~~~~~~

-  Allow to load all files, first X years or last X years in an experiment (`#1133 <https://github.com/ESMValGroup/ESMValCore/pull/1133>`__) `sloosvel <https://github.com/sloosvel>`__
-  Filter tasks earlier (`#1264 <https://github.com/ESMValGroup/ESMValCore/pull/1264>`__) `Javier Vegas-Regidor <https://github.com/jvegreg>`__
-  Added earlier validation for command line arguments (`#1435 <https://github.com/ESMValGroup/ESMValCore/pull/1435>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Remove `profile_diagnostic` from diagnostic settings and increase test coverage of `_task.py` (`#1404 <https://github.com/ESMValGroup/ESMValCore/pull/1404>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Add `output2` to the `product` extra facet of CMIP5 data (`#1514 <https://github.com/ESMValGroup/ESMValCore/pull/1514>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Speed up ESGF search (`#1512 <https://github.com/ESMValGroup/ESMValCore/pull/1512>`__) `Bouwe Andela <https://github.com/bouweandela>`__


.. _changelog-v2-4-0:

v2.4.0
------

Highlights
~~~~~~~~~~

- ESMValCore now has the ability to automatically download missing data from ESGF. For details, see :ref:`Data Retrieval<data-retrieval>`.
- ESMValCore now also can resume an earlier run. This is useful to re-use expensive preprocessor results. For details, see :ref:`Running<running>`.

This release includes

Bug fixes
~~~~~~~~~

-  Crop on the ID-selected region(s) and not on the whole shapefile (`#1151 <https://github.com/ESMValGroup/ESMValCore/pull/1151>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Add 'comment' to list of removed attributes (`#1244 <https://github.com/ESMValGroup/ESMValCore/pull/1244>`__) `Peter Kalverla <https://github.com/Peter9192>`__
-  Speed up multimodel statistics and fix bug in peak computation (`#1301 <https://github.com/ESMValGroup/ESMValCore/pull/1301>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  No longer make plots of provenance (`#1307 <https://github.com/ESMValGroup/ESMValCore/pull/1307>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  No longer embed provenance in output files (`#1306 <https://github.com/ESMValGroup/ESMValCore/pull/1306>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Removed automatic addition of areacello to obs4mips datasets (`#1316 <https://github.com/ESMValGroup/ESMValCore/pull/1316>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Pin docutils <0.17 to fix bullet lists on readthedocs (`#1320 <https://github.com/ESMValGroup/ESMValCore/pull/1320>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Fix obs4MIPs capitalization (`#1328 <https://github.com/ESMValGroup/ESMValCore/pull/1328>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix Python 3.7 tests (`#1330 <https://github.com/ESMValGroup/ESMValCore/pull/1330>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Handle fx variables in `extract_levels` and some time operations (`#1269 <https://github.com/ESMValGroup/ESMValCore/pull/1269>`__) `sloosvel <https://github.com/sloosvel>`__
-  Refactored mask regridding for irregular grids (fixes #772) (`#865 <https://github.com/ESMValGroup/ESMValCore/pull/865>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Fix `da.broadcast_to` call when the fx cube has different shape than target data cube (`#1350 <https://github.com/ESMValGroup/ESMValCore/pull/1350>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Add tests for _aggregate_time_fx (`#1354 <https://github.com/ESMValGroup/ESMValCore/pull/1354>`__) `sloosvel <https://github.com/sloosvel>`__
-  Fix extra facets (`#1360 <https://github.com/ESMValGroup/ESMValCore/pull/1360>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Pin pip!=21.3 to avoid pypa/pip#10573 with editable installs (`#1359 <https://github.com/ESMValGroup/ESMValCore/pull/1359>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Add a custom `date2num` function to deal with changes in cftime (`#1373 <https://github.com/ESMValGroup/ESMValCore/pull/1373>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Removed custom version of `AtmosphereSigmaFactory` (`#1382 <https://github.com/ESMValGroup/ESMValCore/pull/1382>`__) `Manuel Schlund <https://github.com/schlunma>`__

Deprecations
~~~~~~~~~~~~

-  Remove write_netcdf and write_plots from config-user.yml (`#1300 <https://github.com/ESMValGroup/ESMValCore/pull/1300>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Documentation
~~~~~~~~~~~~~

-  Add link to plot directory in index.html (`#1256 <https://github.com/ESMValGroup/ESMValCore/pull/1256>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Work around issue with yapf not following PEP8 (`#1277 <https://github.com/ESMValGroup/ESMValCore/pull/1277>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Update the core development team (`#1278 <https://github.com/ESMValGroup/ESMValCore/pull/1278>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Update the documentation of the provenance interface (`#1305 <https://github.com/ESMValGroup/ESMValCore/pull/1305>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Update version number to first release candidate 2.4.0rc1 (`#1363 <https://github.com/ESMValGroup/ESMValCore/pull/1363>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Update to new ESMValTool logo (`#1374 <https://github.com/ESMValGroup/ESMValCore/pull/1374>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Update version number for third release candidate 2.4.0rc3 (`#1384 <https://github.com/ESMValGroup/ESMValCore/pull/1384>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Update changelog for 2.4.0rc3 (`#1385 <https://github.com/ESMValGroup/ESMValCore/pull/1385>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Update version number to final 2.4.0 release (`#1389 <https://github.com/ESMValGroup/ESMValCore/pull/1389>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Update changelog for 2.4.0 (`#1366 <https://github.com/ESMValGroup/ESMValCore/pull/1366>`__) `Klaus Zimmermann <https://github.com/zklaus>`__

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Add fix for differing latitude coordinate between historical and ssp585 in MPI-ESM1-2-HR r2i1p1f1 (`#1292 <https://github.com/ESMValGroup/ESMValCore/pull/1292>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add fixes for time and latitude coordinate of EC-Earth3 r3i1p1f1 (`#1290 <https://github.com/ESMValGroup/ESMValCore/pull/1290>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Apply latitude fix to all CCSM4 variables (`#1295 <https://github.com/ESMValGroup/ESMValCore/pull/1295>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix lat and lon bounds for FGOALS-g3 mrsos (`#1289 <https://github.com/ESMValGroup/ESMValCore/pull/1289>`__) `Thomas Crocker <https://github.com/thomascrocker>`__
-  Add grid fix for tos in fgoals-f3-l (`#1326 <https://github.com/ESMValGroup/ESMValCore/pull/1326>`__) `sloosvel <https://github.com/sloosvel>`__
-  Add fix for CIESM pr (`#1344 <https://github.com/ESMValGroup/ESMValCore/pull/1344>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix DRS for IPSLCM : split attribute 'freq' into : 'out' and 'freq' (`#1304 <https://github.com/ESMValGroup/ESMValCore/pull/1304>`__) `Stéphane Sénési - work <https://github.com/senesis>`__

CMOR standard
~~~~~~~~~~~~~

-  Remove history attribute from coords (`#1276 <https://github.com/ESMValGroup/ESMValCore/pull/1276>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Increased flexibility of CMOR checks for datasets with generic alevel coordinates (`#1032 <https://github.com/ESMValGroup/ESMValCore/pull/1032>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Automatically fix small deviations in vertical levels (`#1177 <https://github.com/ESMValGroup/ESMValCore/pull/1177>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Adding standard names to the custom tables of the `rlns` and `rsns` variables (`#1386 <https://github.com/ESMValGroup/ESMValCore/pull/1386>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__

Preprocessor
~~~~~~~~~~~~

-  Implemented fully lazy climate_statistics (`#1194 <https://github.com/ESMValGroup/ESMValCore/pull/1194>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Run the multimodel statistics preprocessor last (`#1299 <https://github.com/ESMValGroup/ESMValCore/pull/1299>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Automatic testing
~~~~~~~~~~~~~~~~~

-  Improving test coverage for _task.py (`#514 <https://github.com/ESMValGroup/ESMValCore/pull/514>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Upload coverage to codecov (`#1190 <https://github.com/ESMValGroup/ESMValCore/pull/1190>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Improve codecov status checks (`#1195 <https://github.com/ESMValGroup/ESMValCore/pull/1195>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix curl install in CircleCI (`#1228 <https://github.com/ESMValGroup/ESMValCore/pull/1228>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Drop support for Python 3.6 (`#1200 <https://github.com/ESMValGroup/ESMValCore/pull/1200>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Allow more recent version of `scipy` (`#1182 <https://github.com/ESMValGroup/ESMValCore/pull/1182>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Speed up conda build `conda_build` Circle test by using `mamba` solver via `boa` (and use it for Github Actions test too) (`#1243 <https://github.com/ESMValGroup/ESMValCore/pull/1243>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Fix numpy deprecation warnings (`#1274 <https://github.com/ESMValGroup/ESMValCore/pull/1274>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Unpin upper bound for iris (previously was at <3.0.4)  (`#1275 <https://github.com/ESMValGroup/ESMValCore/pull/1275>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Modernize `conda_install` test on Circle CI by installing from conda-forge with Python 3.9 and change install instructions in documentation (`#1280 <https://github.com/ESMValGroup/ESMValCore/pull/1280>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Run a nightly Github Actions workflow to monitor tests memory per test (configurable for other metrics too) (`#1284 <https://github.com/ESMValGroup/ESMValCore/pull/1284>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Speed up tests of tasks (`#1302 <https://github.com/ESMValGroup/ESMValCore/pull/1302>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix upper case to lower case variables and functions for flake compliance in `tests/unit/preprocessor/_regrid/test_extract_levels.py` (`#1347 <https://github.com/ESMValGroup/ESMValCore/pull/1347>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Cleaned up a bit Github Actions workflows (`#1345 <https://github.com/ESMValGroup/ESMValCore/pull/1345>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Update circleci jobs: renaming tests to more descriptive names and removing conda build test (`#1351 <https://github.com/ESMValGroup/ESMValCore/pull/1351>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Pin iris to latest `>=3.1.0` (`#1341 <https://github.com/ESMValGroup/ESMValCore/pull/1341>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

Installation
~~~~~~~~~~~~

-  Pin esmpy to anything but 8.1.0 since that particular one changes the CPU affinity (`#1310 <https://github.com/ESMValGroup/ESMValCore/pull/1310>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

Improvements
~~~~~~~~~~~~

-  Add a more friendly and useful message when using default config file (`#1233 <https://github.com/ESMValGroup/ESMValCore/pull/1233>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Replace os.walk by glob.glob in data finder (only look for data in the specified locations) (`#1261 <https://github.com/ESMValGroup/ESMValCore/pull/1261>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Machine-specific directories for auxiliary data in the `config-user.yml` file (`#1268 <https://github.com/ESMValGroup/ESMValCore/pull/1268>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Add an option to download missing data from ESGF (`#1217 <https://github.com/ESMValGroup/ESMValCore/pull/1217>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Speed up provenance recording (`#1327 <https://github.com/ESMValGroup/ESMValCore/pull/1327>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Improve results web page (`#1332 <https://github.com/ESMValGroup/ESMValCore/pull/1332>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Move institutes from config-developer.yml to default extra facets config and add wildcard support for extra facets (`#1259 <https://github.com/ESMValGroup/ESMValCore/pull/1259>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add support for re-using preprocessor output from previous runs (`#1321 <https://github.com/ESMValGroup/ESMValCore/pull/1321>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Log fewer messages to screen and hide stack trace for known recipe errors (`#1296 <https://github.com/ESMValGroup/ESMValCore/pull/1296>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Log ESMValCore and ESMValTool versions when running (`#1263 <https://github.com/ESMValGroup/ESMValCore/pull/1263>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Add "grid" as a tag to the output file template for CMIP6 (`#1356 <https://github.com/ESMValGroup/ESMValCore/pull/1356>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Implemented ICON project to read native ICON model output (`#1079 <https://github.com/ESMValGroup/ESMValCore/pull/1079>`__) `Brei Soliño <https://github.com/bsolino>`__


.. _changelog-v2-3-1:

v2.3.1
------

This release includes

Bug fixes
~~~~~~~~~

-  Update config-user.yml template with correct drs entries for CEDA-JASMIN (`#1184 <https://github.com/ESMValGroup/ESMValCore/pull/1184>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Enhancing MIROC5 fix for hfls and evspsbl (`#1192 <https://github.com/ESMValGroup/ESMValCore/pull/1192>`__) `katjaweigel <https://github.com/katjaweigel>`__
-  Fix alignment of daily data with inconsistent calendars in multimodel statistics (`#1212 <https://github.com/ESMValGroup/ESMValCore/pull/1212>`__) `Peter Kalverla <https://github.com/Peter9192>`__
-  Pin cf-units, remove github actions test for Python 3.6 and fix test_access1_0 and test_access1_3 to use cf-units for comparisons (`#1197 <https://github.com/ESMValGroup/ESMValCore/pull/1197>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Fixed search for fx files when no ``mip`` is given (`#1216 <https://github.com/ESMValGroup/ESMValCore/pull/1216>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Make sure climate statistics always returns original dtype (`#1237 <https://github.com/ESMValGroup/ESMValCore/pull/1237>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Bugfix for regional regridding when non-integer range is passed (`#1231 <https://github.com/ESMValGroup/ESMValCore/pull/1231>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Make sure area_statistics preprocessor always returns original dtype (`#1239 <https://github.com/ESMValGroup/ESMValCore/pull/1239>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Add "." (dot) as allowed separation character for the time range group (`#1248 <https://github.com/ESMValGroup/ESMValCore/pull/1248>`__) `Klaus Zimmermann <https://github.com/zklaus>`__

Documentation
~~~~~~~~~~~~~

-  Add a link to the instructions to use pre-installed versions on HPC clusters (`#1186 <https://github.com/ESMValGroup/ESMValCore/pull/1186>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Bugfix release: set version to 2.3.1 (`#1253 <https://github.com/ESMValGroup/ESMValCore/pull/1253>`__) `Klaus Zimmermann <https://github.com/zklaus>`__

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Set circular attribute in MCM-UA-1-0 fix (`#1178 <https://github.com/ESMValGroup/ESMValCore/pull/1178>`__) `sloosvel <https://github.com/sloosvel>`__
-  Fixed time coordinate of MIROC-ESM (`#1188 <https://github.com/ESMValGroup/ESMValCore/pull/1188>`__) `Manuel Schlund <https://github.com/schlunma>`__

Preprocessor
~~~~~~~~~~~~

-  Filter warnings about collapsing multi-model dimension in multimodel statistics preprocessor function (`#1215 <https://github.com/ESMValGroup/ESMValCore/pull/1215>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Remove fx variables before computing multimodel statistics (`#1220 <https://github.com/ESMValGroup/ESMValCore/pull/1220>`__) `sloosvel <https://github.com/sloosvel>`__

Installation
~~~~~~~~~~~~

-  Pin lower bound for iris to 3.0.2 (`#1206 <https://github.com/ESMValGroup/ESMValCore/pull/1206>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Pin `iris<3.0.4` to ensure we still (sort of) support Python 3.6 (`#1252 <https://github.com/ESMValGroup/ESMValCore/pull/1252>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

Improvements
~~~~~~~~~~~~

-  Add test to verify behaviour for scalar height coord for tas in multi-model (`#1209 <https://github.com/ESMValGroup/ESMValCore/pull/1209>`__) `Peter Kalverla <https://github.com/Peter9192>`__
-  Sort missing years in "No input data available for years" message (`#1225 <https://github.com/ESMValGroup/ESMValCore/pull/1225>`__) `Lee de Mora <https://github.com/ledm>`__


.. _changelog-v2-3-0:

v2.3.0
------

This release includes

Bug fixes
~~~~~~~~~

-  Extend preprocessor multi_model_statistics to handle data with "altitude" coordinate (`#1010 <https://github.com/ESMValGroup/ESMValCore/pull/1010>`__) `Axel Lauer <https://github.com/axel-lauer>`__
-  Remove scripts included with CMOR tables (`#1011 <https://github.com/ESMValGroup/ESMValCore/pull/1011>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Avoid side effects in extract_season (`#1019 <https://github.com/ESMValGroup/ESMValCore/pull/1019>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Use nearest scheme to avoid interpolation errors with masked data in regression test (`#1021 <https://github.com/ESMValGroup/ESMValCore/pull/1021>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Move _get_time_bounds from preprocessor._time to cmor.check to avoid circular import with cmor module (`#1037 <https://github.com/ESMValGroup/ESMValCore/pull/1037>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Fix test that makes conda build fail (`#1046 <https://github.com/ESMValGroup/ESMValCore/pull/1046>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Fix 'positive' attribute for rsns/rlns variables (`#1051 <https://github.com/ESMValGroup/ESMValCore/pull/1051>`__) `Lukas Brunner <https://github.com/lukasbrunner>`__
-  Added preprocessor mask_multimodel (`#767 <https://github.com/ESMValGroup/ESMValCore/pull/767>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fix bug when fixing bounds after fixing longitude values (`#1057 <https://github.com/ESMValGroup/ESMValCore/pull/1057>`__) `sloosvel <https://github.com/sloosvel>`__
-  Run conda build parallel AND sequential tests (`#1065 <https://github.com/ESMValGroup/ESMValCore/pull/1065>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Add key to id_prop (`#1071 <https://github.com/ESMValGroup/ESMValCore/pull/1071>`__) `Lukas Brunner <https://github.com/lukasbrunner>`__
-  Fix bounds after reversing coordinate values (`#1061 <https://github.com/ESMValGroup/ESMValCore/pull/1061>`__) `sloosvel <https://github.com/sloosvel>`__
-  Fixed --skip-nonexistent option (`#1093 <https://github.com/ESMValGroup/ESMValCore/pull/1093>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Do not consider CMIP5 variable sit to be the same as sithick from CMIP6 (`#1033 <https://github.com/ESMValGroup/ESMValCore/pull/1033>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Improve finding date range in filenames (enforces separators) (`#1145 <https://github.com/ESMValGroup/ESMValCore/pull/1145>`__) `Stéphane Sénési - work <https://github.com/senesis>`__
-  Review fx handling (`#1147 <https://github.com/ESMValGroup/ESMValCore/pull/1147>`__) `sloosvel <https://github.com/sloosvel>`__
-  Fix lru cache decorator with explicit call to method (`#1172 <https://github.com/ESMValGroup/ESMValCore/pull/1172>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Update _volume.py (`#1174 <https://github.com/ESMValGroup/ESMValCore/pull/1174>`__) `Lee de Mora <https://github.com/ledm>`__

Deprecations
~~~~~~~~~~~~



Documentation
~~~~~~~~~~~~~

-  Final changelog for 2.3.0 (`#1163 <https://github.com/ESMValGroup/ESMValCore/pull/1163>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Set version to 2.3.0 (`#1162 <https://github.com/ESMValGroup/ESMValCore/pull/1162>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Fix documentation build (`#1006 <https://github.com/ESMValGroup/ESMValCore/pull/1006>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add labels required for linking from ESMValTool docs (`#1038 <https://github.com/ESMValGroup/ESMValCore/pull/1038>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Update contribution guidelines (`#1047 <https://github.com/ESMValGroup/ESMValCore/pull/1047>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix basestring references in documentation (`#1106 <https://github.com/ESMValGroup/ESMValCore/pull/1106>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Updated references master to main (`#1132 <https://github.com/ESMValGroup/ESMValCore/pull/1132>`__) `Axel Lauer <https://github.com/axel-lauer>`__
-  Add instructions how to use the central installation at DKRZ-Mistral (`#1155 <https://github.com/ESMValGroup/ESMValCore/pull/1155>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Added fixes for various CMIP5 datasets, variable cl (3-dim cloud fraction) (`#1017 <https://github.com/ESMValGroup/ESMValCore/pull/1017>`__) `Axel Lauer <https://github.com/axel-lauer>`__
-  Added fixes for hybrid level coordinates of CESM2 models (`#882 <https://github.com/ESMValGroup/ESMValCore/pull/882>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Extending LWP fix for CMIP6 models (`#1049 <https://github.com/ESMValGroup/ESMValCore/pull/1049>`__) `Axel Lauer <https://github.com/axel-lauer>`__
-  Add fixes for the net & up radiation variables from ERA5 (`#1052 <https://github.com/ESMValGroup/ESMValCore/pull/1052>`__) `Lukas Brunner <https://github.com/lukasbrunner>`__
-  Add derived variable rsus (`#1053 <https://github.com/ESMValGroup/ESMValCore/pull/1053>`__) `Lukas Brunner <https://github.com/lukasbrunner>`__
-  Supported `mip`-level fixes (`#1095 <https://github.com/ESMValGroup/ESMValCore/pull/1095>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fix erroneous use of `grid_latitude` and `grid_longitude` and cleaned ocean grid fixes (`#1092 <https://github.com/ESMValGroup/ESMValCore/pull/1092>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fix for pr of miroc5 (`#1110 <https://github.com/ESMValGroup/ESMValCore/pull/1110>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Ocean depth fix for cnrm_esm2_1, gfdl_esm4, ipsl_cm6a_lr datasets +  mcm_ua_1_0 (`#1098 <https://github.com/ESMValGroup/ESMValCore/pull/1098>`__) `Tomas Lovato <https://github.com/tomaslovato>`__
-  Fix for uas variable of the MCM_UA_1_0 dataset (`#1102 <https://github.com/ESMValGroup/ESMValCore/pull/1102>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Fixes for sos and siconc of BCC models (`#1090 <https://github.com/ESMValGroup/ESMValCore/pull/1090>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Run fgco2 fix for all CESM2 models (`#1108 <https://github.com/ESMValGroup/ESMValCore/pull/1108>`__) `Lisa Bock <https://github.com/LisaBock>`__
-  Fixes for the siconc variable of CMIP6 models (`#1105 <https://github.com/ESMValGroup/ESMValCore/pull/1105>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Fix wrong sign for land surface flux (`#1113 <https://github.com/ESMValGroup/ESMValCore/pull/1113>`__) `Lisa Bock <https://github.com/LisaBock>`__
-  Fix for pr of EC_EARTH (`#1116 <https://github.com/ESMValGroup/ESMValCore/pull/1116>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__

CMOR standard
~~~~~~~~~~~~~

-  Format cmor related files (`#976 <https://github.com/ESMValGroup/ESMValCore/pull/976>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Check presence of time bounds and guess them if needed (`#849 <https://github.com/ESMValGroup/ESMValCore/pull/849>`__) `sloosvel <https://github.com/sloosvel>`__
-  Add custom variable "tasaga" (`#1118 <https://github.com/ESMValGroup/ESMValCore/pull/1118>`__) `Lisa Bock <https://github.com/LisaBock>`__
-  Find files for CMIP6 DCPP startdates (`#771 <https://github.com/ESMValGroup/ESMValCore/pull/771>`__) `sloosvel <https://github.com/sloosvel>`__

Preprocessor
~~~~~~~~~~~~

-  Update tests for multimodel statistics preprocessor (`#1023 <https://github.com/ESMValGroup/ESMValCore/pull/1023>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Raise in extract_season and extract_month if result is None (`#1041 <https://github.com/ESMValGroup/ESMValCore/pull/1041>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Allow selection of shapes in extract_shape (`#764 <https://github.com/ESMValGroup/ESMValCore/pull/764>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Add option for regional regridding to regrid preprocessor (`#1034 <https://github.com/ESMValGroup/ESMValCore/pull/1034>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Load fx variables as cube cell measures / ancillary variables (`#999 <https://github.com/ESMValGroup/ESMValCore/pull/999>`__) `sloosvel <https://github.com/sloosvel>`__
-  Check horizontal grid before regridding (`#507 <https://github.com/ESMValGroup/ESMValCore/pull/507>`__) `Benjamin Müller <https://github.com/BenMGeo>`__
-  Clip irregular grids (`#245 <https://github.com/ESMValGroup/ESMValCore/pull/245>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Use native iris functions in multi-model statistics (`#1150 <https://github.com/ESMValGroup/ESMValCore/pull/1150>`__) `Peter Kalverla <https://github.com/Peter9192>`__

Notebook API (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~



Automatic testing
~~~~~~~~~~~~~~~~~

-  Report coverage for tests that run on any pull request (`#994 <https://github.com/ESMValGroup/ESMValCore/pull/994>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Install ESMValTool sample data from PyPI (`#998 <https://github.com/ESMValGroup/ESMValCore/pull/998>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Fix tests for multi-processing with spawn method (i.e. macOSX with Python>3.8) (`#1003 <https://github.com/ESMValGroup/ESMValCore/pull/1003>`__) `Barbara Vreede <https://github.com/bvreede>`__
-  Switch to running the Github Action test workflow every 3 hours in single thread mode to observe if Segmentation Faults occur (`#1022 <https://github.com/ESMValGroup/ESMValCore/pull/1022>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Revert to original Github Actions test workflow removing the 3-hourly test run with -n 1 (`#1025 <https://github.com/ESMValGroup/ESMValCore/pull/1025>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Avoid stale cache for multimodel statistics regression tests (`#1030 <https://github.com/ESMValGroup/ESMValCore/pull/1030>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add newer Python versions in OSX to Github Actions (`#1035 <https://github.com/ESMValGroup/ESMValCore/pull/1035>`__) `Barbara Vreede <https://github.com/bvreede>`__
-  Add tests for type annotations with mypy (`#1042 <https://github.com/ESMValGroup/ESMValCore/pull/1042>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Run problematic cmor tests sequentially to avoid segmentation faults on CircleCI (`#1064 <https://github.com/ESMValGroup/ESMValCore/pull/1064>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Test installation of esmvalcore from conda-forge (`#1075 <https://github.com/ESMValGroup/ESMValCore/pull/1075>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Added additional test cases for integration tests of data_finder.py (`#1087 <https://github.com/ESMValGroup/ESMValCore/pull/1087>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Pin cf-units and fix tests (cf-units>=2.1.5) (`#1140 <https://github.com/ESMValGroup/ESMValCore/pull/1140>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Fix failing CircleCI tests (`#1167 <https://github.com/ESMValGroup/ESMValCore/pull/1167>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix test failing due to fx files chosen differently on different OS's (`#1169 <https://github.com/ESMValGroup/ESMValCore/pull/1169>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Compare datetimes instead of strings in _fixes/cmip5/test_access1_X.py (`#1173 <https://github.com/ESMValGroup/ESMValCore/pull/1173>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Pin Python to 3.9 in environment.yml on CircleCI and skip mypy tests in conda build (`#1176 <https://github.com/ESMValGroup/ESMValCore/pull/1176>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Installation
~~~~~~~~~~~~

-  Update yamale to version 3 (`#1059 <https://github.com/ESMValGroup/ESMValCore/pull/1059>`__) `Klaus Zimmermann <https://github.com/zklaus>`__

Improvements
~~~~~~~~~~~~

-  Refactor diagnostics / tags management (`#939 <https://github.com/ESMValGroup/ESMValCore/pull/939>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Support multiple paths in input_dir (`#1000 <https://github.com/ESMValGroup/ESMValCore/pull/1000>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Generate HTML report with recipe output (`#991 <https://github.com/ESMValGroup/ESMValCore/pull/991>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Add timeout to requests.get in _citation.py (`#1091 <https://github.com/ESMValGroup/ESMValCore/pull/1091>`__) `SarahAlidoost <https://github.com/SarahAlidoost>`__
-  Add SYNDA drs for CMIP5 and CMIP6 (closes #582) (`#583 <https://github.com/ESMValGroup/ESMValCore/pull/583>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Add basic support for variable mappings (`#1124 <https://github.com/ESMValGroup/ESMValCore/pull/1124>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Handle IPSL-CM6  (`#1153 <https://github.com/ESMValGroup/ESMValCore/pull/1153>`__) `Stéphane Sénési - work <https://github.com/senesis>`__


.. _changelog-v2-2-0:

v2.2.0
------

Highlights
~~~~~~~~~~

ESMValCore is now using the recently released `Iris 3 <https://scitools-iris.readthedocs.io/en/latest/whatsnew/3.0.html>`__.
We acknowledge that this change may impact your work, as Iris 3 introduces
several changes that are not backward-compatible, but we think that moving forward is the best
decision for the tool in the long term.

This release is also the first one including support for downloading CMIP6 data
using Synda and we have also started supporting Python 3.9. Give it a try!


This release includes

Bug fixes
~~~~~~~~~

-  Fix path settings for DKRZ/Mistral (`#852 <https://github.com/ESMValGroup/ESMValCore/pull/852>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Change logic for calling the diagnostic script to avoid problems with scripts where the executable bit is accidentally set (`#877 <https://github.com/ESMValGroup/ESMValCore/pull/877>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix overwriting in generic level check (`#886 <https://github.com/ESMValGroup/ESMValCore/pull/886>`__) `sloosvel <https://github.com/sloosvel>`__
-  Add double quotes to script args in rerun screen message when using vprof profiling (`#897 <https://github.com/ESMValGroup/ESMValCore/pull/897>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Simplify time handling in multi-model statistics preprocessor (`#685 <https://github.com/ESMValGroup/ESMValCore/pull/685>`__) `Peter Kalverla <https://github.com/Peter9192>`__
-  Fix links to Iris documentation (`#966 <https://github.com/ESMValGroup/ESMValCore/pull/966>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Bugfix: Fix units for MSWEP data (`#986 <https://github.com/ESMValGroup/ESMValCore/pull/986>`__) `Stef Smeets <https://github.com/stefsmeets>`__

Deprecations
~~~~~~~~~~~~

-  Deprecate defining write_plots and write_netcdf in config-user file (`#808 <https://github.com/ESMValGroup/ESMValCore/pull/808>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Documentation
~~~~~~~~~~~~~

-  Fix numbering of steps in release instructions (`#838 <https://github.com/ESMValGroup/ESMValCore/pull/838>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add labels to changelogs of individual versions for easy reference (`#899 <https://github.com/ESMValGroup/ESMValCore/pull/899>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Make CircleCI badge specific to main branch (`#902 <https://github.com/ESMValGroup/ESMValCore/pull/902>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix docker build badge url (`#906 <https://github.com/ESMValGroup/ESMValCore/pull/906>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Update github PR template (`#909 <https://github.com/ESMValGroup/ESMValCore/pull/909>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Refer to ESMValTool GitHub discussions page in the error message (`#900 <https://github.com/ESMValGroup/ESMValCore/pull/900>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Support automatically closing issues (`#922 <https://github.com/ESMValGroup/ESMValCore/pull/922>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix checkboxes in PR template (`#931 <https://github.com/ESMValGroup/ESMValCore/pull/931>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Change in config-user defaults and documentation with new location for esmeval OBS data on JASMIN (`#958 <https://github.com/ESMValGroup/ESMValCore/pull/958>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Update Core Team info (`#942 <https://github.com/ESMValGroup/ESMValCore/pull/942>`__) `Axel Lauer <https://github.com/axel-lauer>`__
-  Update iris documentation URL for sphinx (`#964 <https://github.com/ESMValGroup/ESMValCore/pull/964>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Set version to 2.2.0 (`#977 <https://github.com/ESMValGroup/ESMValCore/pull/977>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Add first draft of v2.2.0 changelog (`#983 <https://github.com/ESMValGroup/ESMValCore/pull/983>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Add checkbox in PR template to assign labels (`#985 <https://github.com/ESMValGroup/ESMValCore/pull/985>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Update install.rst (`#848 <https://github.com/ESMValGroup/ESMValCore/pull/848>`__) `bascrezee <https://github.com/bascrezee>`__
-  Change the order of the publication steps (`#984 <https://github.com/ESMValGroup/ESMValCore/pull/984>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Add instructions how to use esmvaltool from HPC central installations (`#841 <https://github.com/ESMValGroup/ESMValCore/pull/841>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Fixing unit for derived variable rsnstcsnorm to prevent overcorrection2 (`#846 <https://github.com/ESMValGroup/ESMValCore/pull/846>`__) `katjaweigel <https://github.com/katjaweigel>`__
-  Cmip6 fix awi cm 1 1 mr (`#822 <https://github.com/ESMValGroup/ESMValCore/pull/822>`__) `mwjury <https://github.com/mwjury>`__
-  Cmip6 fix ec earth3 veg (`#836 <https://github.com/ESMValGroup/ESMValCore/pull/836>`__) `mwjury <https://github.com/mwjury>`__
-  Changed latitude longitude fix from Tas to AllVars. (`#916 <https://github.com/ESMValGroup/ESMValCore/pull/916>`__) `katjaweigel <https://github.com/katjaweigel>`__
-  Fix for precipitation (pr) to use ERA5-Land cmorizer (`#879 <https://github.com/ESMValGroup/ESMValCore/pull/879>`__) `katjaweigel <https://github.com/katjaweigel>`__
-  Cmip6 fix ec earth3 (`#837 <https://github.com/ESMValGroup/ESMValCore/pull/837>`__) `mwjury <https://github.com/mwjury>`__
-  Cmip6_fix_fgoals_f3_l_Amon_time_bnds (`#831 <https://github.com/ESMValGroup/ESMValCore/pull/831>`__) `mwjury <https://github.com/mwjury>`__
-  Fix for FGOALS-f3-L sftlf (`#667 <https://github.com/ESMValGroup/ESMValCore/pull/667>`__) `mwjury <https://github.com/mwjury>`__
-  Improve ACCESS-CM2 and ACCESS-ESM1-5 fixes and add CIESM and CESM2-WACCM-FV2 fixes for cl, clw and cli (`#635 <https://github.com/ESMValGroup/ESMValCore/pull/635>`__) `Axel Lauer <https://github.com/axel-lauer>`__
-  Add  fixes for cl, cli, clw and tas for several CMIP6 models (`#955 <https://github.com/ESMValGroup/ESMValCore/pull/955>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Dataset fixes for MSWEP (`#969 <https://github.com/ESMValGroup/ESMValCore/pull/969>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Dataset fixes for: ACCESS-ESM1-5, CanESM5, CanESM5 for carbon cycle (`#947 <https://github.com/ESMValGroup/ESMValCore/pull/947>`__) `Bettina Gier <https://github.com/bettina-gier>`__
-  Fixes for KIOST-ESM (CMIP6) (`#904 <https://github.com/ESMValGroup/ESMValCore/pull/904>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__
-  Fixes for AWI-ESM-1-1-LR (CMIP6, piControl) (`#911 <https://github.com/ESMValGroup/ESMValCore/pull/911>`__) `Rémi Kazeroni <https://github.com/remi-kazeroni>`__

CMOR standard
~~~~~~~~~~~~~

-  CMOR check generic level coordinates in CMIP6 (`#598 <https://github.com/ESMValGroup/ESMValCore/pull/598>`__) `sloosvel <https://github.com/sloosvel>`__
-  Update CMIP6 tables to 6.9.33 (`#919 <https://github.com/ESMValGroup/ESMValCore/pull/919>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Adding custom variables for tas uncertainty (`#924 <https://github.com/ESMValGroup/ESMValCore/pull/924>`__) `Lisa Bock <https://github.com/LisaBock>`__
-  Remove monotonicity coordinate check for unstructured grids (`#965 <https://github.com/ESMValGroup/ESMValCore/pull/965>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__

Preprocessor
~~~~~~~~~~~~

-  Added clip_start_end_year preprocessor (`#796 <https://github.com/ESMValGroup/ESMValCore/pull/796>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Add support for downloading CMIP6 data with Synda (`#699 <https://github.com/ESMValGroup/ESMValCore/pull/699>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add multimodel tests using real data (`#856 <https://github.com/ESMValGroup/ESMValCore/pull/856>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Add plev/altitude conversion to extract_levels (`#892 <https://github.com/ESMValGroup/ESMValCore/pull/892>`__) `Axel Lauer <https://github.com/axel-lauer>`__
-  Add possibility of custom season extraction. (`#247 <https://github.com/ESMValGroup/ESMValCore/pull/247>`__) `mwjury <https://github.com/mwjury>`__
-  Adding the ability to derive xch4  (`#783 <https://github.com/ESMValGroup/ESMValCore/pull/783>`__) `Birgit Hassler <https://github.com/hb326>`__
-  Add preprocessor function to resample time and compute x-hourly statistics (`#696 <https://github.com/ESMValGroup/ESMValCore/pull/696>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Fix duplication in preprocessors DEFAULT_ORDER introduced in #696 (`#973 <https://github.com/ESMValGroup/ESMValCore/pull/973>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Use consistent precision in multi-model statistics calculation and update reference data for tests (`#941 <https://github.com/ESMValGroup/ESMValCore/pull/941>`__) `Peter Kalverla <https://github.com/Peter9192>`__
-  Refactor multi-model statistics code to facilitate ensemble stats and lazy evaluation (`#949 <https://github.com/ESMValGroup/ESMValCore/pull/949>`__) `Peter Kalverla <https://github.com/Peter9192>`__
-  Add option to exclude input cubes in output of multimodel statistics to solve an issue introduced by #949 (`#978 <https://github.com/ESMValGroup/ESMValCore/pull/978>`__) `Peter Kalverla <https://github.com/Peter9192>`__


Automatic testing
~~~~~~~~~~~~~~~~~

-  Pin cftime>=1.3.0 to have newer string formatting in and fix two tests (`#878 <https://github.com/ESMValGroup/ESMValCore/pull/878>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Switched miniconda conda setup hooks for Github Actions workflows (`#873 <https://github.com/ESMValGroup/ESMValCore/pull/873>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Add test for latest version resolver (`#874 <https://github.com/ESMValGroup/ESMValCore/pull/874>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Update codacy coverage reporter to fix coverage (`#905 <https://github.com/ESMValGroup/ESMValCore/pull/905>`__) `Niels Drost <https://github.com/nielsdrost>`__
-  Avoid hardcoded year in tests and add improvement to plev test case (`#921 <https://github.com/ESMValGroup/ESMValCore/pull/921>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Pin scipy to less than 1.6.0 until ESMValGroup/ESMValCore/issues/927 gets resolved (`#928 <https://github.com/ESMValGroup/ESMValCore/pull/928>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Github Actions: change time when conda install test runs (`#930 <https://github.com/ESMValGroup/ESMValCore/pull/930>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Remove redundant test line from test_utils.py (`#935 <https://github.com/ESMValGroup/ESMValCore/pull/935>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Removed netCDF4 package from integration tests of fixes (`#938 <https://github.com/ESMValGroup/ESMValCore/pull/938>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Use new conda environment for installing ESMValCore in Docker containers (`#951 <https://github.com/ESMValGroup/ESMValCore/pull/951>`__) `Bouwe Andela <https://github.com/bouweandela>`__

Notebook API (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Implement importable config object in experimental API submodule (`#868 <https://github.com/ESMValGroup/ESMValCore/pull/868>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Add loading and running recipes to the notebook API (`#907 <https://github.com/ESMValGroup/ESMValCore/pull/907>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Add displaying and loading of recipe output to the notebook API (`#957 <https://github.com/ESMValGroup/ESMValCore/pull/957>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Add functionality to run single diagnostic task to notebook API (`#962 <https://github.com/ESMValGroup/ESMValCore/pull/962>`__) `Stef Smeets <https://github.com/stefsmeets>`__

Improvements
~~~~~~~~~~~~

-  Create CODEOWNERS file (`#809 <https://github.com/ESMValGroup/ESMValCore/pull/809>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Remove code needed for Python <3.6 (`#844 <https://github.com/ESMValGroup/ESMValCore/pull/844>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add requests as a dependency (`#850 <https://github.com/ESMValGroup/ESMValCore/pull/850>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Pin Python to less than 3.9 (`#870 <https://github.com/ESMValGroup/ESMValCore/pull/870>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Remove numba dependency (`#880 <https://github.com/ESMValGroup/ESMValCore/pull/880>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Add Listing and finding recipes to the experimental notebook API (`#901 <https://github.com/ESMValGroup/ESMValCore/pull/901>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Skip variables that don't have dataset or additional_dataset keys (`#860 <https://github.com/ESMValGroup/ESMValCore/pull/860>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Refactor logging configuration (`#933 <https://github.com/ESMValGroup/ESMValCore/pull/933>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Xco2 derivation (`#913 <https://github.com/ESMValGroup/ESMValCore/pull/913>`__) `Bettina Gier <https://github.com/bettina-gier>`__
-  Working environment for Python 3.9 (pin to !=3.9.0) (`#885 <https://github.com/ESMValGroup/ESMValCore/pull/885>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Print source file when using config get_config_user command (`#960 <https://github.com/ESMValGroup/ESMValCore/pull/960>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Switch to Iris 3 (`#819 <https://github.com/ESMValGroup/ESMValCore/pull/819>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Refactor tasks (`#959 <https://github.com/ESMValGroup/ESMValCore/pull/959>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Restore task summary in debug log after #959 (`#981 <https://github.com/ESMValGroup/ESMValCore/pull/981>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Pin pre-commit hooks (`#974 <https://github.com/ESMValGroup/ESMValCore/pull/974>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Improve error messages when data is missing (`#917 <https://github.com/ESMValGroup/ESMValCore/pull/917>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Set remove_preproc_dir to false in default config-user (`#979 <https://github.com/ESMValGroup/ESMValCore/pull/979>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Move fiona to be installed from conda forge (`#987 <https://github.com/ESMValGroup/ESMValCore/pull/987>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Re-added fiona in setup.py (`#990 <https://github.com/ESMValGroup/ESMValCore/pull/990>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

.. _changelog-v2-1-0:

v2.1.0
------

This release includes

Bug fixes
~~~~~~~~~

-  Set unit=1 if anomalies are standardized (`#727 <https://github.com/ESMValGroup/ESMValCore/pull/727>`__) `bascrezee <https://github.com/bascrezee>`__
-  Fix crash for FGOALS-g2 variables without longitude coordinate (`#729 <https://github.com/ESMValGroup/ESMValCore/pull/729>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Improve variable alias management (`#595 <https://github.com/ESMValGroup/ESMValCore/pull/595>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Fix area_statistics fx files loading (`#798 <https://github.com/ESMValGroup/ESMValCore/pull/798>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Fix units after derivation (`#754 <https://github.com/ESMValGroup/ESMValCore/pull/754>`__) `Manuel Schlund <https://github.com/schlunma>`__

Documentation
~~~~~~~~~~~~~

-  Update v2.0.0 release notes with final additions (`#722 <https://github.com/ESMValGroup/ESMValCore/pull/722>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Update package description in setup.py (`#725 <https://github.com/ESMValGroup/ESMValCore/pull/725>`__) `Mattia Righi <https://github.com/mattiarighi>`__
-  Add installation instructions for pip installation (`#735 <https://github.com/ESMValGroup/ESMValCore/pull/735>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Improve config-user documentation (`#740 <https://github.com/ESMValGroup/ESMValCore/pull/740>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Update the zenodo file with contributors (`#807 <https://github.com/ESMValGroup/ESMValCore/pull/807>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Improve command line run documentation (`#721 <https://github.com/ESMValGroup/ESMValCore/pull/721>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Update the zenodo file with contributors (continued) (`#810 <https://github.com/ESMValGroup/ESMValCore/pull/810>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

Improvements
~~~~~~~~~~~~

-  Reduce size of docker image (`#723 <https://github.com/ESMValGroup/ESMValCore/pull/723>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Add 'test' extra to installation, used by docker development tag (`#733 <https://github.com/ESMValGroup/ESMValCore/pull/733>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Correct dockerhub link (`#736 <https://github.com/ESMValGroup/ESMValCore/pull/736>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Create action-install-from-pypi.yml (`#734 <https://github.com/ESMValGroup/ESMValCore/pull/734>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Add pre-commit for linting/formatting (`#766 <https://github.com/ESMValGroup/ESMValCore/pull/766>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Run tests in parallel and when building conda package (`#745 <https://github.com/ESMValGroup/ESMValCore/pull/745>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Readable exclude pattern for pre-commit (`#770 <https://github.com/ESMValGroup/ESMValCore/pull/770>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Github Actions Tests (`#732 <https://github.com/ESMValGroup/ESMValCore/pull/732>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Remove isort setup to fix formatting conflict with yapf (`#778 <https://github.com/ESMValGroup/ESMValCore/pull/778>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Fix yapf-isort import formatting conflict (Fixes #777) (`#784 <https://github.com/ESMValGroup/ESMValCore/pull/784>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Sorted output for `esmvaltool recipes list` (`#790 <https://github.com/ESMValGroup/ESMValCore/pull/790>`__) `Stef Smeets <https://github.com/stefsmeets>`__
-  Replace vmprof with vprof (`#780 <https://github.com/ESMValGroup/ESMValCore/pull/780>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Update CMIP6 tables to 6.9.32 (`#706 <https://github.com/ESMValGroup/ESMValCore/pull/706>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Default config-user path now set in config-user read function (`#791 <https://github.com/ESMValGroup/ESMValCore/pull/791>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Add custom variable lweGrace (`#692 <https://github.com/ESMValGroup/ESMValCore/pull/692>`__) `bascrezee <https://github.com/bascrezee>`__
- Create Github Actions workflow to build and deploy on Test PyPi and PyPi (`#820 <https://github.com/ESMValGroup/ESMValCore/pull/820>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
- Build and publish the esmvalcore package to conda via Github Actions workflow (`#825 <https://github.com/ESMValGroup/ESMValCore/pull/825>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Fix cmip6 models (`#629 <https://github.com/ESMValGroup/ESMValCore/pull/629>`__) `npgillett <https://github.com/npgillett>`__
-  Fix siconca variable in EC-Earth3 and EC-Earth3-Veg models in amip simulation (`#702 <https://github.com/ESMValGroup/ESMValCore/pull/702>`__) `Evgenia Galytska <https://github.com/egalytska>`__

Preprocessor
~~~~~~~~~~~~

-  Move cmor_check_data to early in preprocessing chain (`#743 <https://github.com/ESMValGroup/ESMValCore/pull/743>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add RMS iris analysis operator to statistics preprocessor functions (`#747 <https://github.com/ESMValGroup/ESMValCore/pull/747>`__) `Pep Cos <https://github.com/pcosbsc>`__
-  Add surface chlorophyll concentration as a derived variable (`#720 <https://github.com/ESMValGroup/ESMValCore/pull/720>`__) `sloosvel <https://github.com/sloosvel>`__
-  Use dask to reduce memory consumption of extract_levels for masked data (`#776 <https://github.com/ESMValGroup/ESMValCore/pull/776>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

.. _changelog-v2-0-0:

v2.0.0
------

This release includes

Bug fixes
~~~~~~~~~

-  Fixed derivation of co2s (`#594 <https://github.com/ESMValGroup/ESMValCore/pull/594>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Padding while cropping needs to stay within sane bounds for shapefiles that span the whole Earth (`#626 <https://github.com/ESMValGroup/ESMValCore/pull/626>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Fix concatenation of a single cube (`#655 <https://github.com/ESMValGroup/ESMValCore/pull/655>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix mask fx dict handling not to fail if empty list in values (`#661 <https://github.com/ESMValGroup/ESMValCore/pull/661>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Preserve metadata during anomalies computation when using iris cubes difference (`#652 <https://github.com/ESMValGroup/ESMValCore/pull/652>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Avoid crashing when there is directory 'esmvaltool' in the current working directory (`#672 <https://github.com/ESMValGroup/ESMValCore/pull/672>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Solve bug in ACCESS1 dataset fix for calendar.  (`#671 <https://github.com/ESMValGroup/ESMValCore/pull/671>`__) `Peter Kalverla <https://github.com/Peter9192>`__
-  Fix the syntax for adding multiple ensemble members from the same dataset (`#678 <https://github.com/ESMValGroup/ESMValCore/pull/678>`__) `SarahAlidoost <https://github.com/SarahAlidoost>`__
-  Fix bug that made preprocessor with fx files fail in rare cases (`#670 <https://github.com/ESMValGroup/ESMValCore/pull/670>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Add support for string coordinates (`#657 <https://github.com/ESMValGroup/ESMValCore/pull/657>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Fixed the shape extraction to account for wraparound shapefile coords (`#319 <https://github.com/ESMValGroup/ESMValCore/pull/319>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Fixed bug in time weights calculation (`#695 <https://github.com/ESMValGroup/ESMValCore/pull/695>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fix diagnostic filter (`#713 <https://github.com/ESMValGroup/ESMValCore/pull/713>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__

Documentation
~~~~~~~~~~~~~

-  Add pandas as a requirement for building the documentation (`#607 <https://github.com/ESMValGroup/ESMValCore/pull/607>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Document default order in which preprocessor functions are applied (`#633 <https://github.com/ESMValGroup/ESMValCore/pull/633>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add pointers about data loading and CF standards to documentation (`#571 <https://github.com/ESMValGroup/ESMValCore/pull/571>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Config file populated with site-specific data paths examples (`#619 <https://github.com/ESMValGroup/ESMValCore/pull/619>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Update Codacy badges (`#643 <https://github.com/ESMValGroup/ESMValCore/pull/643>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Update copyright info on readthedocs (`#668 <https://github.com/ESMValGroup/ESMValCore/pull/668>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Updated references to documentation (now docs.esmvaltool.org) (`#675 <https://github.com/ESMValGroup/ESMValCore/pull/675>`__) `Axel Lauer <https://github.com/axel-lauer>`__
-  Add all European grants to Zenodo (`#680 <https://github.com/ESMValGroup/ESMValCore/pull/680>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Update Sphinx to v3 or later (`#683 <https://github.com/ESMValGroup/ESMValCore/pull/683>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Increase version to 2.0.0 and add release notes (`#691 <https://github.com/ESMValGroup/ESMValCore/pull/691>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Update setup.py and README.md for use on PyPI (`#693 <https://github.com/ESMValGroup/ESMValCore/pull/693>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Suggested Documentation changes (`#690 <https://github.com/ESMValGroup/ESMValCore/pull/690>`__) `Steve Smith <https://github.com/ssmithClimate>`__

Improvements
~~~~~~~~~~~~

-  Reduce the size of conda package (`#606 <https://github.com/ESMValGroup/ESMValCore/pull/606>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add a few unit tests for DiagnosticTask (`#613 <https://github.com/ESMValGroup/ESMValCore/pull/613>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Make ncl or R tests not fail if package not installed (`#610 <https://github.com/ESMValGroup/ESMValCore/pull/610>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Pin flake8<3.8.0 (`#623 <https://github.com/ESMValGroup/ESMValCore/pull/623>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Log warnings for likely errors in provenance record (`#592 <https://github.com/ESMValGroup/ESMValCore/pull/592>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Unpin flake8 (`#646 <https://github.com/ESMValGroup/ESMValCore/pull/646>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  More flexible native6 default DRS (`#645 <https://github.com/ESMValGroup/ESMValCore/pull/645>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Try to use the same python for running diagnostics as for esmvaltool (`#656 <https://github.com/ESMValGroup/ESMValCore/pull/656>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix test for lower python version and add note on lxml (`#659 <https://github.com/ESMValGroup/ESMValCore/pull/659>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Added 1m deep average soil moisture variable (`#664 <https://github.com/ESMValGroup/ESMValCore/pull/664>`__) `bascrezee <https://github.com/bascrezee>`__
-  Update docker recipe (`#603 <https://github.com/ESMValGroup/ESMValCore/pull/603>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Improve command line interface (`#605 <https://github.com/ESMValGroup/ESMValCore/pull/605>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Remove utils directory (`#697 <https://github.com/ESMValGroup/ESMValCore/pull/697>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Avoid pytest version that crashes (`#707 <https://github.com/ESMValGroup/ESMValCore/pull/707>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Options arg in read_config_user_file now optional (`#716 <https://github.com/ESMValGroup/ESMValCore/pull/716>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Produce a readable warning if ancestors are a string instead of a list. (`#711 <https://github.com/ESMValGroup/ESMValCore/pull/711>`__) `katjaweigel <https://github.com/katjaweigel>`__
-  Pin Yamale to v2 (`#718 <https://github.com/ESMValGroup/ESMValCore/pull/718>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Expanded cmor public API (`#714 <https://github.com/ESMValGroup/ESMValCore/pull/714>`__) `Manuel Schlund <https://github.com/schlunma>`__

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Added various fixes for hybrid height coordinates (`#562 <https://github.com/ESMValGroup/ESMValCore/pull/562>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Extended fix for cl-like variables of CESM2 models (`#604 <https://github.com/ESMValGroup/ESMValCore/pull/604>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Added fix to convert "geopotential" to "geopotential height" for ERA5 (`#640 <https://github.com/ESMValGroup/ESMValCore/pull/640>`__) `Evgenia Galytska <https://github.com/egalytska>`__
-  Do not fix longitude values if they are too far from valid range (`#636 <https://github.com/ESMValGroup/ESMValCore/pull/636>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__

Preprocessor
~~~~~~~~~~~~

-  Implemented concatenation of cubes with derived coordinates (`#546 <https://github.com/ESMValGroup/ESMValCore/pull/546>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Fix derived variable ctotal calculation depending on project and standard name (`#620 <https://github.com/ESMValGroup/ESMValCore/pull/620>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  State of the art FX variables handling without preprocessing (`#557 <https://github.com/ESMValGroup/ESMValCore/pull/557>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Add max, min and std operators to multimodel (`#602 <https://github.com/ESMValGroup/ESMValCore/pull/602>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Added preprocessor to extract amplitude of cycles (`#597 <https://github.com/ESMValGroup/ESMValCore/pull/597>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Overhaul concatenation and allow for correct concatenation of multiple overlapping datasets (`#615 <https://github.com/ESMValGroup/ESMValCore/pull/615>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Change volume stats to handle and output masked array result (`#618 <https://github.com/ESMValGroup/ESMValCore/pull/618>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Area_weights for cordex in area_statistics (`#631 <https://github.com/ESMValGroup/ESMValCore/pull/631>`__) `mwjury <https://github.com/mwjury>`__
-  Accept cubes as input in multimodel (`#637 <https://github.com/ESMValGroup/ESMValCore/pull/637>`__) `sloosvel <https://github.com/sloosvel>`__
-  Make multimodel work correctly with yearly data (`#677 <https://github.com/ESMValGroup/ESMValCore/pull/677>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Optimize time weights in time preprocessor for climate statistics (`#684 <https://github.com/ESMValGroup/ESMValCore/pull/684>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Add percentiles to multi-model stats (`#679 <https://github.com/ESMValGroup/ESMValCore/pull/679>`__) `Peter Kalverla <https://github.com/Peter9192>`__

.. _changelog-v2-0-0b9:

v2.0.0b9
--------

This release includes

Bug fixes
~~~~~~~~~

-  Cast dtype float32 to output from zonal and meridional area preprocessors (`#581 <https://github.com/ESMValGroup/ESMValCore/pull/581>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__

Improvements
~~~~~~~~~~~~

-  Unpin on Python<3.8 for conda package (run) (`#570 <https://github.com/ESMValGroup/ESMValCore/pull/570>`__) `Valeriu Predoi <https://github.com/valeriupredoi>`__
-  Update pytest installation marker (`#572 <https://github.com/ESMValGroup/ESMValCore/pull/572>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Remove vmrh2o (`#573 <https://github.com/ESMValGroup/ESMValCore/pull/573>`__) `Mattia Righi <https://github.com/mattiarighi>`__
-  Restructure documentation (`#575 <https://github.com/ESMValGroup/ESMValCore/pull/575>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Fix mask in land variables for CCSM4 (`#579 <https://github.com/ESMValGroup/ESMValCore/pull/579>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Fix derive scripts wrt required method (`#585 <https://github.com/ESMValGroup/ESMValCore/pull/585>`__) `Klaus Zimmermann <https://github.com/zklaus>`__
-  Check coordinates do not have repeated standard names (`#558 <https://github.com/ESMValGroup/ESMValCore/pull/558>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Added derivation script for co2s (`#587 <https://github.com/ESMValGroup/ESMValCore/pull/587>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Adapted custom co2s table to match CMIP6 version (`#588 <https://github.com/ESMValGroup/ESMValCore/pull/588>`__) `Manuel Schlund <https://github.com/schlunma>`__
-  Increase version to v2.0.0b9 (`#593 <https://github.com/ESMValGroup/ESMValCore/pull/593>`__) `Bouwe Andela <https://github.com/bouweandela>`__
-  Add a method to save citation information (`#402 <https://github.com/ESMValGroup/ESMValCore/pull/402>`__) `SarahAlidoost <https://github.com/SarahAlidoost>`__

For older releases, see the release notes on https://github.com/ESMValGroup/ESMValCore/releases.
