.. _changelog:

Changelog
=========

.. _changelog-v2-11-0:

v2.11.0
-------
Highlights

- Performance improvements have been made to many preprocessors:

  - Preprocessors :func:`esmvalcore.preprocessor.mask_landsea`,
    :func:`esmvalcore.preprocessor.mask_landseaice`,
    :func:`esmvalcore.preprocessor.mask_glaciated`,
    :func:`esmvalcore.preprocessor.extract_levels` are now lazy

- Several new preprocessors have been added:

  - :func:`esmvalcore.preprocessor.local_solar_time`
  - :func:`esmvalcore.preprocessor.distance_metrics`
  - :func:`esmvalcore.preprocessor.histogram`

- NEW TREND: First time release manager shout-outs!

  - This is the first ESMValTool release managed by the Met Office! We want to
    shout this out - and for all future first time release managers to
    shout-out - to celebrate the growing, thriving ESMValTool community.

This release includes

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Allow contiguous representation of extracted regions (:pull:`2230`) by :user:`rebeccaherman1`

   - The preprocessor function :func:`esmvalcore.preprocessor.extract_region`
     no longer automatically maps the extracted :class:`iris.cube.Cube` to the
     0-360 degrees longitude domain. If you need this behaviour, use
     ``cube.intersection(longitude=(0., 360.))`` in your Python code after
     extracting the region. There is no possibility to restore the previous
     behaviour from a recipe.

-  Use ``iris.FUTURE.save_split_attrs = True`` to remove iris warning (:pull:`2398`) by :user:`schlunma`

   - Since `v3.8.0`_, Iris explicitly distinguishes between local and global
     netCDF attributes. ESMValCore adopted this behavior with v2.11.0. With
     this change, attributes are written as local attributes by default, unless
     they already existed as global attributes or belong to a special list of
     global attributes (in which case attributes are written as global
     attributes). See :class:`iris.cube.CubeAttrsDict` for details.

.. _v3.8.0: https://scitools-iris.readthedocs.io/en/stable/whatsnew/3.8.html#v3-8-29-feb-2024

Deprecations
~~~~~~~~~~~~

-  Refactor regridding (:pull:`2231`) by :user:`schlunma`

   - This PR deprecated two regridding schemes, which will be removed with
     ESMValCore v2.13.0:

     - ``unstructured_nearest``: Please use the scheme ``nearest`` instead.
       This is an exact replacement for data on unstructured grids. ESMValCore
       is now able to determine the most suitable regridding scheme based on
       the input data.
     - ``linear_extrapolate``: Please use a generic scheme with
       ``reference: iris.analysis:Linear`` and
       ``extrapolation_mode: extrapolate`` instead.

-  Allow deprecated regridding scheme ``linear_extrapolate`` in recipe checks (:pull:`2324`) by :user:`schlunma`
-  Allow deprecated regridding scheme ``unstructured_nearest`` in recipe checks (:pull:`2336`) by :user:`schlunma`

Bug fixes
~~~~~~~~~

-  Do not overwrite facets from recipe with CMOR table facets for derived variables (:pull:`2255`) by :user:`bouweandela`
-  Fix error message in variable definition check (:pull:`2313`) by :user:`enekomartinmartinez`
-  Unify dtype handling of preprocessors (:pull:`2393`) by :user:`schlunma`
-  Fix bug in ``_rechunk_aux_factory_dependencies`` (:pull:`2428`) by :user:`ehogan`
-  Avoid loading entire files into memory when downloading from ESGF (:pull:`2434`) by :user:`bouweandela`
-  Preserve cube attribute global vs local when concatenating (:pull:`2449`) by :user:`bouweandela`

CMOR standard
~~~~~~~~~~~~~

-  Also read default custom CMOR tables if custom location is specified (:pull:`2279`) by :user:`schlunma`
-  Add custom CMOR table for total cloud water (tcw) (:pull:`2277`) by :user:`axel-lauer`
-  Add height for sfcWindmax in MPI HighRes models (:pull:`2292`) by :user:`malininae`
-  Fixed ``positive`` attribute in custom rtnt table (:pull:`2367`) by :user:`schlunma`
-  Fix ``positive`` attributes in custom CMOR variables (:pull:`2380`) by :user:`schlunma`
-  Log CMOR check and generic fix output to separate file (:pull:`2361`) by :user:`schlunma`

Computational performance improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  More lazy fixes and preprocessing functions (:pull:`2325`) by :user:`bouweandela`
-  Made preprocessors :func:`esmvalcore.preprocessor.mask_landsea`, :func:`esmvalcore.preprocessor.mask_landseaice` and :func:`esmvalcore.preprocessor.mask_glaciated` lazy  (:pull:`2268`) by :user:`joergbenke`
-  More lazy :func:`esmvalcore.preprocessor.extract_levels` preprocessor function (:pull:`2120`) by :user:`bouweandela`
-  Use lazy weights for :func:`esmvalcore.preprocessor.climate_statistics` and :func:`esmvalcore.preprocessor.axis_statistics` (:pull:`2346`) by :user:`schlunma`
-  Fixed potential memory leak in :func:`esmvalcore.preprocessor.local_solar_time` (:pull:`2356`) by :user:`schlunma`
-  Cache regridding weights if possible (:pull:`2344`) by :user:`schlunma`
-  Implement lazy area weights (:pull:`2354`) by :user:`schlunma`
-  Avoid large chunks in :func:`esmvalcore.preprocessor.climate_statistics` preprocessor function with `period='full'` (:pull:`2404`) by :user:`bouweandela`
-  Load data only once for ESMPy regridders (:pull:`2418`) by :user:`bouweandela`

Documentation
~~~~~~~~~~~~~

-  Use short links in changelog (:pull:`2287`) by :user:`bouweandela`
-  National Computing Infrastructure (NCI), Site specific configuration (:pull:`2281`) by :user:`rbeucher`
-  Update :func:`esmvalcore.preprocessor.multi_model_statistics` doc with latest changes (new operators, etc.) (:pull:`2321`) by :user:`schlunma`
-  Fix Codacy badge (:pull:`2382`) by :user:`bouweandela`
-  Change 'mean' to 'percentile' in doc strings of preprocessor statistics (:pull:`2327`) by :user:`lukruh`
-  Fixed typo in doc about weighted statistics (:pull:`2387`) by :user:`schlunma`

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Fixing missing height 2m coordinates in GFDL-CM4 and KIOST-ESM (:pull:`2294`) by :user:`Karen-A-Garcia`
-  Added fix for wrong units of ``clt`` for CIESM and FIO-ESM-2-0 (:pull:`2330`) by :user:`schlunma`
-  Cmip6 gfdl_cm4: fix tas height fix to work for concatenated scenarios (:pull:`2332`) by :user:`mwjury`
-  Cordex GERICS REMO2015  lon differences above 10e-4 (:pull:`2334`) by :user:`mwjury`
-  Download ICON grid without locking (:pull:`2359`) by :user:`bouweandela`
-  Added ICON fixes for hfls and hfss (:pull:`2360`) by :user:`diegokam`
-  Added ICON fix for rtnt (:pull:`2366`) by :user:`diegokam`
-  Expanded ICON extra facets (:pull:`2379`) by :user:`schlunma`
-  Add 10m height coordinate to SfcWind GFDL-CM4 instead of 2m height (:pull:`2385`) by :user:`Karen-A-Garcia`
-  Cordex wrf381p: fix tas,tasmax,tasmin height (:pull:`2333`) by :user:`mwjury`
-  Several minor fixes needed for marine BGC data. (:pull:`2110`) by :user:`ledm`

Installation
~~~~~~~~~~~~

-  Pin pandas yet again avoid new ``2.2.1`` as well (:pull:`2353`) by :user:`valeriupredoi`
-  Update Iris pin to avoid using versions with memory issues (:pull:`2408`) by :user:`chrisbillowsMO`
-  Pin esmpy <8.6.0 (:pull:`2402`) by :user:`valeriupredoi`
-  Pin numpy<2.0.0 to avoid pulling 2.0.0rcX (:pull:`2415`) by :user:`valeriupredoi`
-  Add support for Python=3.12 (:pull:`2228`) by :user:`valeriupredoi`

Preprocessor
~~~~~~~~~~~~

-  New preprocessor: :func:`esmvalcore.preprocessor.local_solar_time` (:pull:`2258`) by :user:`schlunma`
-  Read derived variables from other MIP tables (:pull:`2256`) by :user:`bouweandela`
-  Added special unit conversion m -> DU for total column ozone (toz) (:pull:`2270`) by :user:`schlunma`
-  Allow cubes as input for :func:`esmvalcore.preprocessor.bias` preprocessor (:pull:`2183`) by :user:`schlunma`
-  Add normalization with statistics to many statistics preprocessors (:pull:`2189`) by :user:`schlunma`
-  Adding sfcWind derivation from uas and vas  (:pull:`2242`) by :user:`malininae`
-  Update interval check in resample_hours (:pull:`2362`) by :user:`axel-lauer`
-  Broadcast properly ``cell_measures`` when using :func:`esmvalcore.preprocessor.extract_shape` with ``decomposed: True`` (:pull:`2348`) by :user:`sloosvel`
-  Compute volume from ``cell_area`` if available (:pull:`2318`) by :user:`enekomartinmartinez`
-  Do not expand wildcards for datasets of derived variables where not all input variables are available (:pull:`2374`) by :user:`schlunma`
-  Modernize :func:`esmvalcore.preprocessor.regrid_time` and allow setting a common calendar for decadal, yearly, and monthly data (:pull:`2311`) by :user:`schlunma`
-  Added unstructured linear regridding (:pull:`2350`) by :user:`schlunma`
-  Add preprocessors :func:`esmvalcore.preprocessor.distance_metrics` and :func:`esmvalcore.preprocessor.histogram` (:pull:`2299`) by :user:`schlunma`

Automatic testing
~~~~~~~~~~~~~~~~~

-  Increase resources for testing installation from conda-forge (:pull:`2297`) by :user:`bouweandela`
-  Pin pandas to avoid broken ``round`` function (:pull:`2305`) by :user:`schlunma`
-  Remove team reviewers from conda lock generation workflow in Github Actions (:pull:`2307`) by :user:`valeriupredoi`
-  Remove mocking from tests in ``tests/unit/preprocessor/_regrid/test_extract_point.py`` (:pull:`2193`) by :user:`ehogan`
-  Pin ``pytest-mypy`` plugin to ``>=0.10.3`` comply with new ``pytest==8`` (:pull:`2315`) by :user:`valeriupredoi`
-  Fix regridding test for unstructured nearest regridding on OSX (:pull:`2319`) by :user:`schlunma`
-  Fix flaky regrid test by clearing LRU cache after each test (:pull:`2322`) by :user:`valeriupredoi`
-  Xfail ``tests/integration/cmor/_fixes/test_common.py::test_cl_hybrid_height_coord_fix_metadata`` while Iris folk fix behaviour (:pull:`2363`) by :user:`valeriupredoi`
-  Update codacy reporter orb to latest version (:pull:`2388`) by :user:`valeriupredoi`
-  Add calls to ``conda list`` in Github Action test workflows to inspect environment (:pull:`2391`) by :user:`valeriupredoi`
-  Pin pandas yet again :panda_face: ``test_icon`` fails again with pandas=2.2.2 (:pull:`2394`) by :user:`valeriupredoi`
-  Fixed units of cl test data (necessary since iris>=3.8.0) (:pull:`2403`) by :user:`schlunma`

Improvements
~~~~~~~~~~~~

-  Show files of supplementary variables explicitly in log (:pull:`2303`) by :user:`schlunma`
-  Remove warning about logging in to ESGF (:pull:`2326`) by :user:`bouweandela`
-  Do not read ``~/.esmvaltool/config-user.yml`` if ``--config-file`` is used (:pull:`2309`) by :user:`schlunma`
-  Support loading ICON grid from ICON rootpath (:pull:`2337`) by :user:`schlunma`
-  Handle warnings about invalid units for iris>=3.8 (:pull:`2378`) by :user:`schlunma`
-  Added note on how to access ``index.html`` on remote server (:pull:`2276`) by :user:`schlunma`
-  Remove custom fix for concatenation of aux factories now that bug in iris is solved (:pull:`2392`) by :user:`schlunma`
-  Ignored iris warnings about global attributes (:pull:`2400`) by :user:`schlunma`
-  Add native6, OBS6 and RAWOBS rootpaths to metoffice config-user.yml template, and remove temporary dir (:pull:`2432`) by :user:`alistairsellar`

.. _changelog-v2-10-0:

v2.10.0
-------
Highlights

-  All statistics preprocessors support the same operators and have a common
   :ref:`documentation <stat_preprocs>`. In addition, arbitrary keyword arguments
   for the statistical operation can be directly given to the preprocessor.

-  The output webpage generated by the tool now looks better and provides
   methods to select and filter the output.

-  Improved computational efficiency:

   -  Automatic rechunking between preprocessor steps to keep the
      `graph size smaller <https://docs.dask.org/en/latest/best-practices.html#avoid-very-large-graphs>`_
      and the `chunk size optimal <https://blog.dask.org/2021/11/02/choosing-dask-chunk-sizes>`__.
   -  Reduce the size of the dask graph created by :func:`esmvalcore.preprocessor.anomalies`.
   -  Preprocessors :func:`esmvalcore.preprocessor.mask_above_threshold`,
      :func:`esmvalcore.preprocessor.mask_below_threshold`,
      :func:`esmvalcore.preprocessor.mask_inside_range`,
      :func:`esmvalcore.preprocessor.mask_outside_range` are now lazy.
   -  Lazy coordinates bounds are no longer loaded into memory by the CMOR checks and fixes.

This release includes

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Remove the deprecated option ``use_legacy_supplementaries`` (:pull:`2202`) by :user:`bouweandela`

   - The recommended upgrade procedure is to remove ``use_legacy_supplementaries`` from config-user.yml
     (if it was there) and remove any mention of ``fx_variables`` from the recipe. If automatically defining
     the required supplementary variables does not work, define them in the variable or
     (``additional_``) ``datasets`` section as described in :ref:`supplementary_variables`.

-  Use smarter (units-aware) weights (:pull:`2139`) by :user:`schlunma`

   - Some preprocessors handle units better. For details, see the pull request.

-  Removed deprecated configuration option ``offline`` (:pull:`2213`) by :user:`schlunma`

   - In :ref:`changelog-v2-8-0`, we replaced the old ``offline`` configuration option. From this version on, it stops working.
     Please refer to :ref:`changelog-v2-8-0` for upgrade instructions.

-  Fix issue with CORDEX datasets requiring different dataset tags for downloads and fixes (:pull:`2066`) by :user:`ljoakim`

   - Due to the different facets for CORDEX datasets, there was an inconsistency in the fixing mechanism.
     This change requires changes to existing recipes that use CORDEX datasets. Please refer to the pull request for detailed update instructions.

-  For the following changes, no user change is necessary

   -  Remove deprecated way of calling :func:`~esmvalcore.cmor.table.read_cmor_tables` (:pull:`2201`) by :user:`bouweandela`

   -  Remove deprecated callback argument from preprocessor ``load`` function (:pull:`2207`) by :user:`bouweandela`

   -  Remove deprecated preprocessor function `cleanup` (:pull:`2215`) by :user:`bouweandela`

Deprecations
~~~~~~~~~~~~

-  Clearly separate fixes and CMOR checks (:pull:`2157`) by :user:`schlunma`
-  Added new operators for statistics preprocessor (e.g., ``'percentile'``) and allowed passing additional arguments (:pull:`2191`) by :user:`schlunma`

   - This harmonizes the operators for all statistics preprocessors. From this version, the new names can be used; the old arguments will stop working from
     version 2.12.0. Please refer to :ref:`stat_preprocs` for a detailed description.

Bug fixes
~~~~~~~~~

-  Re-add correctly region-extracted cell measures and ancillary variables after :ref:`extract_region` (:pull:`2166`) by :user:`valeriupredoi`, :user:`schlunma`
-  Fix sorting of datasets

   -  Fix sorting of ensemble members in :func:`~esmvalcore.dataset.datasets_to_recipe` (:pull:`2095`) by :user:`bouweandela`
   -  Fix a problem with sorting datasets that have a mix of facet types (:pull:`2238`) by :user:`bouweandela`
   -  Avoid a crash if dataset has supplementary variables (:pull:`2198`) by :user:`bouweandela`

CMOR standard
~~~~~~~~~~~~~

-  ERA5 on-the-fly CMORizer: changed sign of variables ``evspsbl`` and ``evspsblpot`` (:pull:`2115`) by :user:`katjaweigel`
-  Add ``ch4`` surface custom cmor table entry (:pull:`2168`) by :user:`hb326`
-  Add CMIP3 institutes names used at NCI (:pull:`2152`) by :user:`rbeucher`
-  Added :func:`~esmvalcore.cmor.fixes.get_time_bounds` and :func:`~esmvalcore.cmor.fixes.get_next_month` to public API (:pull:`2214`) by :user:`schlunma`
-  Improve concatenation checks

   -  Relax concatenation checks for ``--check_level=relax`` and ``--check_level=ignore`` (:pull:`2144`) by :user:`sloosvel`
   -  Fix ``concatenate`` preprocessor function (:pull:`2240`) by :user:`bouweandela`
   -  Fix time overlap handling in concatenation (:pull:`2247`) by :user:`zklaus`

Computational performance improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Make :ref:`threshold_masking` preprocessors lazy  (:pull:`2169`) by :user:`joergbenke`

   -  Restored usage of numpy in `_mask_with_shp` (:pull:`2209`) by :user:`joergbenke`
-  Do not realize lazy coordinate bounds in CMOR check (:pull:`2146`) by :user:`sloosvel`
-  Rechunk between preprocessor steps (:pull:`2205`) by :user:`bouweandela`
-  Reduce the size of the dask graph created by the ``anomalies`` preprocessor function (:pull:`2200`) by :user:`bouweandela`

Documentation
~~~~~~~~~~~~~

-  Add reference to release v2.9.0 in the changelog (:pull:`2130`) by :user:`remi-kazeroni`
-  Add merge instructions to release instructions (:pull:`2131`) by :user:`zklaus`
-  Update `mamba` before building environment during Readthedocs build (:pull:`2149`) by :user:`valeriupredoi`
-  Ensure compatible zstandard and zstd versions for .conda support (:pull:`2204`) by :user:`zklaus`
-  Remove outdated documentation (:pull:`2210`) by :user:`bouweandela`
-  Remove meercode badge from README because their API is broken (:pull:`2224`) by :user:`valeriupredoi`
-  Correct usage help text of version command (:pull:`2232`) by :user:`jfrost-mo`
-  Add ``navigation_with_keys: False`` to ``html_theme_options`` in Readthedocs ``conf.py`` (:pull:`2245`) by :user:`valeriupredoi`
-  Replace squarey badge with roundy shield for Anaconda sticker in README (:pull:`2233`, :pull:`2260`) by :user:`valeriupredoi`

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Updated doc about fixes and added type hints to fix functions (:pull:`2160`) by :user:`schlunma`

Installation
~~~~~~~~~~~~

-  Clean-up how pins are written in conda environment file (:pull:`2125`) by :user:`valeriupredoi`
-  Use importlib.metadata instead of deprecated pkg_resources (:pull:`2096`) by :user:`bouweandela`
-  Pin shapely to >=2.0 (:pull:`2075`) by :user:`valeriupredoi`
-  Pin Python to <3.12 in conda environment (:pull:`2272`) by :user:`bouweandela`

Preprocessor
~~~~~~~~~~~~

-  Improve preprocessor output sorting code (:pull:`2111`) by :user:`bouweandela`
-  Preprocess datasets in the same order as they are listed in the recipe (:pull:`2103`) by :user:`bouweandela`

Automatic testing
~~~~~~~~~~~~~~~~~

-  [Github Actions] Compress all bash shell setters into one default option per workflow (:pull:`2126`) by :user:`valeriupredoi`
-  [Github Actions] Fix Monitor Tests Github Action (:pull:`2135`) by :user:`valeriupredoi`
-  [condalock] update conda lock file (:pull:`2141`) by :user:`valeriupredoi`
-  [Condalock] make sure mamba/conda are at latest version by forcing a pinned mamba install (:pull:`2136`) by :user:`valeriupredoi`
-  Update code coverage orbs (:pull:`2206`) by :user:`bouweandela`
-  Revisit the comment-triggered Github Actions test (:pull:`2243`) by :user:`valeriupredoi`
-  Remove workflow that runs Github Actions tests from PR comment (:pull:`2244`) by :user:`valeriupredoi`

Improvements
~~~~~~~~~~~~

-  Merge v2.9.x into main (:pull:`2128`) by :user:`schlunma`
-  Fix typo in citation file (:pull:`2182`) by :user:`bouweandela`
-  Cleaned and extended function that extracts datetimes from paths (:pull:`2181`) by :user:`schlunma`
-  Add file encoding (and some read modes) at open file step (:pull:`2219`) by :user:`valeriupredoi`
-  Check type of argument passed to :func:`~esmvalcore.cmor.table.read_cmor_tables` (:pull:`2217`) by :user:`valeriupredoi`
-  Dynamic HTML output for monitoring (:pull:`2062`) by :user:`bsolino`
-  Use PyPI's trusted publishers authentication (:pull:`2269`) by :user:`valeriupredoi`

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
your experiences, good and bad, with this new feature in :discussion:`1763`.

This release includes

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Remove deprecated configuration options (:pull:`2056`) by :user:`bouweandela`

   - The module ``esmvalcore.experimental.config`` has been removed.
     To upgrade, import the module from :mod:`esmvalcore.config` instead.

   - The module ``esmvalcore._config`` has been removed.
     To upgrade, use :mod:`esmvalcore.config` instead.

   - The methods ``esmvalcore.config.Session.to_config_user`` and ``esmvalcore.config.Session.from_config_user`` have been removed.
     To upgrade, use :obj:`esmvalcore.config.Session` to access the configuration values directly.

Bug fixes
~~~~~~~~~

-  Respect ``ignore_warnings`` settings from the :ref:`project configuration <filterwarnings_config-developer>` in :func:`esmvalcore.dataset.Dataset.load` (:pull:`2046`) by :user:`schlunma`
-  Fixed usage of custom location for :ref:`custom CMOR tables <custom_cmor_tables>` (:pull:`2052`) by :user:`schlunma`
-  Fix issue with writing index.html when :ref:`running a recipe <running>` with ``--resume-from`` (:pull:`2055`) by :user:`bouweandela`
-  Fixed bug in ICON CMORizer that lead to shifted time coordinates (:pull:`2038`) by :user:`schlunma`
-  Include ``-`` in allowed characters for bibtex references (:pull:`2097`) by :user:`alistairsellar`
-  Do not raise an exception if the requested version of a file is not available for all matching files on ESGF (:pull:`2105`) by :user:`bouweandela`

Computational performance improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Add support for :ref:`configuring Dask distributed <config-dask>` (:pull:`2049`, :pull:`2122`) by :user:`bouweandela`
-  Make :func:`esmvalcore.preprocessor.extract_levels` lazy (:pull:`1761`) by :user:`bouweandela`
-  Lazy implementation of :func:`esmvalcore.preprocessor.multi_model_statistics` and :func:`esmvalcore.preprocessor.ensemble_statistics` (:pull:`968` and :pull:`2087`) by :user:`Peter9192`
-  Avoid realizing data in preprocessor function :func:`esmvalcore.preprocessor.concatenate` when cubes overlap (:pull:`2109`) by :user:`bouweandela`

Documentation
~~~~~~~~~~~~~

-  Remove unneeded sphinxcontrib extension (:pull:`2047`) by :user:`valeriupredoi`
-  Show ESMValTool logo on `PyPI webpage <https://pypi.org/project/ESMValCore/>`__ (:pull:`2065`) by :user:`valeriupredoi`
-  Fix gitter badge in README (:pull:`2118`) by :user:`remi-kazeroni`
-  Add changelog for v2.9.0 (:pull:`2088` and :pull:`2123`) by :user:`bouweandela`

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Pass the :obj:`esmvalcore.config.Session` to fixes (:pull:`1988`) by :user:`schlunma`
-  ICON: Allowed specifying vertical grid information in recipe (:pull:`2067`) by :user:`schlunma`
-  Allow specifying ``raw_units`` for CESM2, EMAC, and ICON CMORizers (:pull:`2043`) by :user:`schlunma`
-  ICON: allow specifying horizontal grid file in recipe/extra facets (:pull:`2078`) by :user:`schlunma`
-  Fix tas/tos CMIP6: FIO, KACE, MIROC, IITM (:pull:`2061`) by :user:`pepcos`
-  Add fix for EC-Earth3-Veg tos calendar (:pull:`2100`) by :user:`bouweandela`
-  Correct GISS-E2-1-G ``tos`` units (:pull:`2099`) by :user:`bouweandela`

Installation
~~~~~~~~~~~~

-  Drop support for Python 3.8 (:pull:`2053`) by :user:`bouweandela`
-  Add python 3.11 to Github Actions package (conda and PyPI) installation tests (:pull:`2083`) by :user:`valeriupredoi`
-  Remove ``with_mypy`` or ``with-mypy`` optional tool for prospector (:pull:`2108`) by :user:`valeriupredoi`

Preprocessor
~~~~~~~~~~~~

-  Added ``period='hourly'`` for :func:`esmvalcore.preprocessor.climate_statistics` and :func:`esmvalcore.preprocessor.anomalies` (:pull:`2068`) by :user:`schlunma`
-  Support IPCC AR6 regions in :func:`esmvalcore.preprocessor.extract_shape` (:pull:`2008`) by :user:`schlunma`


.. _changelog-v2-8-1:

v2.8.1
------
Highlights
~~~~~~~~~~
This release adds support for Python 3.11 and includes several bugfixes.

This release includes:

Bug fixes
~~~~~~~~~

-  Pin numpy !=1.24.3 (:pull:`2011`) by :user:`valeriupredoi`
-  Fix a bug in recording provenance for the ``mask_multimodel`` preprocessor (:pull:`1984`) by :user:`schlunma`
-  Fix ICON hourly data rounding issues (:pull:`2022`) by :user:`BauerJul`
-  Use the default SSL context when using the ``extract_location`` preprocessor (:pull:`2023`) by :user:`ehogan`
-  Make time-related CMOR fixes work with time dimensions `time1`, `time2`, `time3` (:pull:`1971`) by :user:`schlunma`
-  Always create a cache directory for storing ICON grid files (:pull:`2030`) by :user:`schlunma`
-  Fixed altitude <--> pressure level conversion for masked arrays in the ``extract_levels`` preprocessor (:pull:`1999`) by :user:`schlunma`
-  Allowed ignoring of scalar time coordinates in the ``multi_model_statistics`` preprocessor (:pull:`1961`) by :user:`schlunma`

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Add support for hourly ICON data (:pull:`1990`) by :user:`BauerJul`
-  Fix areacello in BCC-CSM2-MR (:pull:`1993`) by :user:`remi-kazeroni`

Installation
~~~~~~~~~~~~

-  Add support for Python=3.11 (:pull:`1832`) by :user:`valeriupredoi`
-  Modernize conda lock file creation workflow with mamba, Mambaforge etc (:pull:`2027`) by :user:`valeriupredoi`
-  Pin `libnetcdf!=4.9.1` (:pull:`2072`) by :user:`remi-kazeroni`

Documentation
~~~~~~~~~~~~~
-  Add changelog for v2.8.1 (:pull:`2079`) by :user:`bouweandela`

Automatic testing
~~~~~~~~~~~~~~~~~

-  Use mocked `geopy.geocoders.Nominatim` to avoid `ReadTimeoutError` (:pull:`2005`) by :user:`schlunma`
-  Update pre-commit hooks (:pull:`2020`) by :user:`bouweandela`


.. _changelog-v2-8-0:


v2.8.0
------
Highlights
~~~~~~~~~~

-  ESMValCore now supports wildcards in recipes and offers improved support for
   ancillary variables and dataset versioning thanks to contributions by
   :user:`bouweandela`. For details, see
   :ref:`Automatically populating a recipe with all available datasets <dataset_wildcards>`
   and :ref:`Defining supplementary variables <supplementary_variables>`.
-  Support for CORDEX datasets in a rotated pole coordinate system has been
   added by :user:`sloosvel`.
-  Native :ref:`ICON <read_icon>` output is now made UGRID-compliant
   on-the-fly to unlock the use of more sophisticated regridding algorithms,
   thanks to :user:`schlunma`.
-  The Python API has been extended with the addition of three
   modules: :mod:`esmvalcore.config`, :mod:`esmvalcore.dataset`, and
   :mod:`esmvalcore.local`, all these features courtesy of
   :user:`bouweandela`. For details, see our new
   example :doc:`example-notebooks`.
-  The preprocessor :func:`~esmvalcore.preprocessor.multi_model_statistics`
   has been extended to support more use-cases thanks to contributions by
   :user:`schlunma`. For details, see
   :ref:`Multi-model statistics <multi-model statistics>`.

This release includes:

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Please read the descriptions of the linked pull requests for detailed upgrade instructions.

-  The algorithm for automatically defining the ancillary variables and cell
   measures has been improved (:pull:`1609`) by :user:`bouweandela`.
   If this does not work as expected, more examples of how to adapt your recipes
   are given
   `here <https://github.com/ESMValGroup/ESMValCore/pull/1609#Backward-incompatible-changes>`__
   and in the corresponding sections of the
   :ref:`recipe documentation <supplementary_variables>` and the
   :ref:`preprocessor documentation <preprocessors_using_supplementary_variables>`.
-  Remove deprecated features scheduled for removal in v2.8.0 or earlier
   (:pull:`1826`) by :user:`schlunma`.
   Removed ``esmvalcore.iris_helpers.var_name_constraint`` (has been deprecated
   in v2.6.0; please use :class:`iris.NameConstraint` with the keyword argument
   ``var_name`` instead) and the option ``always_use_ne_mask`` for
   :func:`esmvalcore.preprocessor.mask_landsea` (has been deprecated in v2.5.0;
   the same behavior can now be achieved by specifying ``supplementary_variables``.
-  No files will be found if a non-existent version of a dataset is specified
   (:pull:`1835`) by :user:`bouweandela`. If a ``version`` of a
   dataset is specified in the recipe, the tool will now search for exactly that
   version, instead of simply using the latest version. Therefore, it is
   necessary to make sure that the version number in the directory tree matches
   with the version number in the recipe to find the files.
-  The default filename template for obs4MIPs has been updated to better match
   filenames used in this project in (:pull:`1866`) by :user:`bouweandela`. This
   may cause issues if you are storing all the files for obs4MIPs in a
   directory with no subdirectories per dataset.

Deprecations
~~~~~~~~~~~~
Please read the descriptions of the linked pull requests for detailed upgrade instructions.

-  Various configuration related options that are now available through
   :mod:`esmvalcore.config` have been deprecated (:pull:`1769`) by :user:`bouweandela`.
-  The ``fx_variables`` preprocessor argument and related features have been
   deprecated (:pull:`1609`) by :user:`bouweandela`.
   See :pull:`1609#Deprecations` for more information.
-  Combined ``offline`` and ``always_search_esgf`` into a single option ``search_esgf``
   (:pull:`1935`)
   :user:`schlunma`. The configuration
   option/command line argument ``offline`` has been deprecated in favor of
   ``search_esgf``. The previous ``offline: true`` is now ``search_esgf: never``
   (the default); the previous ``offline: false`` is now
   ``search_esgf: when_missing``. More details on how to adapt your workflow
   regarding these new options are given in :pull:`1935` and the
   `documentation <https://docs.esmvaltool.org/projects/ESMValCore/en/latest/quickstart/configure.html?highlight=search_esgf#user-configuration-file>`__.
-  :func:`esmvalcore.preprocessor.cleanup` has been deprecated (:pull:`1949`)
   :user:`schlunma`. Please do not use this
   anymore in the recipe (it is not necessary).

Python API
~~~~~~~~~~

-  Support searching ESGF for a specific version of a file and add :obj:`esmvalcore.esgf.ESGFFile.facets` (:pull:`1822`) by :user:`bouweandela`
-  Fix issues with searching for files on ESGF (:pull:`1863`) by :user:`bouweandela`
-  Move the :mod:`esmvalcore.experimental.config` module to  :mod:`esmvalcore.config` (:pull:`1769`) by :user:`bouweandela`
-  Add :mod:`esmvalcore.local`, a module to search data on the local filesystem (:pull:`#1835`) by :user:`bouweandela`
-  Add :mod:`esmvalcore.dataset` module (:pull:`1877`) by :user:`bouweandela`

Bug fixes
~~~~~~~~~

-  Import from :mod:`esmvalcore.config` in the :mod:`esmvalcore.experimental` module (:pull:`1816`) by :user:`bouweandela`
-  Added scalar coords of input cubes to output of esmpy_regrid (:pull:`1811`) by :user:`schlunma`
-  Fix severe bug in :func:`esmvalcore.preprocessor.mask_fillvalues` (:pull:`1823`) by :user:`schlunma`
-  Fix LWP of ICON on-the-fly CMORizer (:pull:`1839`) by :user:`schlunma`
-  Fixed issue in irregular regridding regarding scalar coordinates (:pull:`1845`) by :user:`schlunma`
-  Update product attributes and `metadata.yml` with cube metadata before saving files (:pull:`1837`) by :user:`schlunma`
-  Remove an extra space character from a filename (:pull:`1883`) by :user:`bouweandela`
-  Improve resilience of ESGF search (:pull:`1869`) by :user:`bouweandela`
-  Fix issue with no files found if timerange start/end differs in length (:pull:`1880`) by :user:`bouweandela`
-  Add `driver` and `sub_experiment` tags to generate dataset aliases (:pull:`1886`) by :user:`sloosvel`
-  Fixed time points of native CESM2 output (:pull:`1772`) by :user:`schlunma`
-  Fix type hints for Python versions < 3.10 (:pull:`1897`) by :user:`bouweandela`
-  Fixed `set_range_in_0_360` for dask arrays (:pull:`1919`) by :user:`schlunma`
-  Made equalized attributes in concatenated cubes consistent across runs (:pull:`1783`) by :user:`schlunma`
-  Fix issue with reading dates from files (:pull:`1936`) by :user:`bouweandela`
-  Add institute name used on ESGF for CMIP5 CanAM4, CanCM4, and CanESM2 (:pull:`1937`) by :user:`bouweandela`
-  Fix issue where data was not loaded and saved (:pull:`1962`) by :user:`bouweandela`
-  Fix type hints for Python 3.8 (:pull:`1795`) by :user:`bouweandela`
-  Update the institute facet of the CSIRO-Mk3L-1-2 model (:pull:`1966`) by :user:`remi-kazeroni`
-  Fixed race condition that may result in errors in :func:`esmvalcore.preprocessor.cleanup` (:pull:`1949`) by :user:`schlunma`
-  Update notebook so it uses supplementaries instead of ancillaries (:pull:`1945`) by :user:`bouweandela`

Documentation
~~~~~~~~~~~~~

-  Fix anaconda badge in README (:pull:`1759`) by :user:`valeriupredoi`
-  Fix mistake in the documentation of :obj:`esmvalcore.esgf.find_files` (:pull:`1784`) by :user:`bouweandela`
-  Support linking to "stable" ESMValTool version on readthedocs (:pull:`1608`) by :user:`bouweandela`
-  Updated ICON doc with information on usage of extract_levels preprocessor (:pull:`1903`) by :user:`schlunma`
-  Add changelog for latest released version v2.7.1 (:pull:`1905`) by :user:`valeriupredoi`
-  Update `preprocessor.rst` due to renaming of NCEP dataset to NCEP-NCAR-R1 (:pull:`1908`) by :user:`hb326`
-  Replace timerange nested lists in docs with overview table (:pull:`1940`) by :user:`zklaus`
-  Updated section "backward compatibility" in `contributing.rst` (:pull:`1918`) by :user:`axel-lauer`
-  Add link to ESMValTool release procedure steps (:pull:`1957`) by :user:`remi-kazeroni`
-  Synchronize documentation table of contents with ESMValTool (:pull:`1958`) by :user:`bouweandela`

Improvements
~~~~~~~~~~~~

-  Support wildcards in the recipe and improve support for ancillary variables and dataset versioning (:pull:`1609`) by :user:`bouweandela`. More details on how to adapt your recipes are given in the corresponding pull request description and in the corresponding sections of the `recipe documentation <https://docs.esmvaltool.org/projects/esmvalcore/en/latest/recipe/overview.html#defining-supplementary-variables-ancillary-variables-and-cell-measures>`__ and the `preprocessor documentation <https://docs.esmvaltool.org/projects/esmvalcore/en/latest/recipe/preprocessor.html#preprocessors-using-supplementary-variables>`__.
-  Create a session directory with suffix "-1", "-2", etc if it already exists (:pull:`1818`) by :user:`bouweandela`
-  Message for users when they use esmvaltool executable from esmvalcore only (:pull:`1831`) by :user:`valeriupredoi`
-  Order recipe output in index.html (:pull:`1899`) by :user:`bouweandela`
-  Improve reading facets from ESGF search results (:pull:`1920`) by :user:`bouweandela`

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Fix rotated coordinate grids and `tas` and `pr` for CORDEX datasets (:pull:`1765`) by :user:`sloosvel`
-  Made ICON output UGRID-compliant (on-the-fly) (:pull:`1664`) by :user:`schlunma`
-  Fix automatic download of ICON grid file and make ICON UGRIDization optional (`default: true`) (:pull:`1922`) by :user:`schlunma`
-  Add siconc fixes for EC-Earth3-Veg and EC-Earth3-Veg-LR models (:pull:`1771`) by :user:`egalytska`
-  Fix siconc in KIOST-ESM (:pull:`1829`) by :user:`LisaBock`
-  Extension of ERA5 CMORizer (variable cl) (:pull:`1850`) by :user:`axel-lauer`
-  Add standard variable names for EMAC (:pull:`1853`) by :user:`FranziskaWinterstein`
-  Fix for FGOALS-f3-L clt (:pull:`1928`) by :user:`LisaBock`

Installation
~~~~~~~~~~~~

-  Add all deps to the conda-forge environment and suppress installing and reinstalling deps with pip at readthedocs builds (:pull:`1786`) by :user:`valeriupredoi`
-  Pin netCDF4<1.6.1 (:pull:`1805`) by :user:`bouweandela`
-  Unpin NetCF4 (:pull:`1814`) by :user:`valeriupredoi`
-  Unpin flake8 (:pull:`1820`) by :user:`valeriupredoi`
-  Add iris-esmf-regrid as a dependency (:pull:`1809`) by :user:`sloosvel`
-  Pin esmpy<8.4 (:pull:`1871`) by :user:`zklaus`
-  Update esmpy import for ESMF v8.4.0 (:pull:`1876`) by :user:`bouweandela`

Preprocessor
~~~~~~~~~~~~
-  Allow :func:`esmvalcore.preprocessor.multi_model_statistics` on cubes with arbitrary dimensions  (:pull:`1808`) by :user:`schlunma`
-  Smarter removal of coordinate metadata in :func:`esmvalcore.preprocessor.multi_model_statistics` preprocessor (:pull:`1813`) by :user:`schlunma`
-  Allowed usage of :func:`esmvalcore.preprocessor.multi_model_statistics` on single cubes/products (:pull:`1849`) by :user:`schlunma`
-  Allowed usage of :func:`esmvalcore.preprocessor.multi_model_statistics` on cubes with identical ``name()`` and ``units`` (but e.g. different long_name) (:pull:`1921`) by :user:`schlunma`
-  Allowed ignoring scalar coordinates in :func:`esmvalcore.preprocessor.multi_model_statistics` (:pull:`1934`) by :user:`schlunma`
-  Refactored :func:`esmvalcore.preprocessor.regrid` and removed unnecessary code not needed anymore due to new iris version (:pull:`1898`) by :user:`schlunma`
-  Do not realise coordinates during CMOR check (:pull:`1912`) by :user:`sloosvel`
-  Make :func:`esmvalcore.preprocessor.extract_volume` work with closed and mixed intervals and allow nearest value selection (:pull:`1930`) by :user:`sloosvel`

Release
~~~~~~~
-  Changelog for `v2.8.0rc1` (:pull:`1952`) by :user:`remi-kazeroni`
-  Increase version number for ESMValCore `v2.8.0rc1` (:pull:`1955`) by :user:`remi-kazeroni`
-  Changelog for `v2.8.0rc2` (:pull:`1959`) by :user:`remi-kazeroni`
-  Increase version number for ESMValCore `v2.8.0rc2` (:pull:`1973`) by :user:`remi-kazeroni`
-  Changelog for `v2.8.0` (:pull:`1978`) by :user:`remi-kazeroni`
-  Increase version number for ESMValCore `v2.8.0` (:pull:`1983`) by :user:`remi-kazeroni`

Automatic testing
~~~~~~~~~~~~~~~~~

-  Set implicit optional to true in `mypy` config to avert side effects and test fails from new mypy version (:pull:`1790`) by :user:`valeriupredoi`
-  Remove duplicate `implicit_optional = True` line in ``setup.cfg`` (:pull:`1791`) by :user:`valeriupredoi`
-  Fix failing test due to missing sample data (:pull:`1797`) by :user:`bouweandela`
-  Remove outdated cmor_table facet from data finder tests (:pull:`1798`) by :user:`bouweandela`
-  Modernize tests for :func:`esmvalcore.preprocessor.save` (:pull:`1799`) by :user:`bouweandela`
-  No more sequential tests since SegFaults were not noticed anymore (:pull:`1819`) by :user:`valeriupredoi`
-  Update pre-commit configuration (:pull:`1821`) by :user:`bouweandela`
-  Updated URL of ICON grid file used for testing (:pull:`1914`) by :user:`schlunma`

Variable Derivation
~~~~~~~~~~~~~~~~~~~

-  Add derivation of sea ice extent (:pull:`1695`) by :user:`sloosvel`


.. _changelog-v2-7-1:


v2.7.1
------
Highlights
~~~~~~~~~~

This is a bugfix release where we unpin `cf-units` to allow the latest `iris=3.4.0` to be installed. It also includes an update to the default configuration used when searching the ESGF for files, to account for a recent change of the CEDA ESGF index node hostname. The changelog contains only changes that were made to the ``main`` branch.

Installation
~~~~~~~~~~~~

- Set the version number on the development branches to one minor version more than the previous release (:pull:`1854`) by :user:`bouweandela`
- Unpin cf-units (:pull:`1770`) by :user:`bouweandela`

Bug fixes
~~~~~~~~~

- Improve error handling if an esgf index node is offline (:pull:`1834`) by :user:`bouweandela`

Automatic testing
~~~~~~~~~~~~~~~~~

- Removed unnecessary test that fails with iris 3.4.0 (:pull:`1846`) by :user:`schlunma`
- Update CEDA ESGF index node hostname (:pull:`1838`) by :user:`valeriupredoi`


.. _changelog-v2-7-0:


v2.7.0
------
Highlights
~~~~~~~~~~

-  We have a new preprocessor function called `'rolling_window_statistics' <https://docs.esmvaltool.org/projects/ESMValCore/en/latest/recipe/preprocessor.html#rolling-window-statistics>`__ implemented by :user:`malininae`
-  We have improved the support for native models, refactored native model fixes by adding common base class `NativeDatasetFix`, changed default DRS for reading native ICON output, and added tests for input/output filenames for `ICON <https://docs.esmvaltool.org/projects/ESMValCore/en/latest/quickstart/find_data.html#icon>`__ and `EMAC <https://docs.esmvaltool.org/projects/ESMValCore/en/latest/quickstart/find_data.html#emac>`__ on-the-fly CMORizer, all these features courtesy of :user:`schlunma`
-  Performance of preprocessor functions that use time dimensions has been sped up by **two orders of magnitude** thanks to contributions by :user:`bouweandela`

This release includes:

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Change default DRS for reading native ICON output (:pull:`1705`) by :user:`schlunma`

Bug fixes
~~~~~~~~~

-  Add support for regions stored as MultiPolygon to extract_shape preprocessor (:pull:`1670`) by :user:`bouweandela`
-  Fixed type annotations for Python 3.8 (:pull:`1700`) by :user:`schlunma`
-  Core `_io.concatenate()` may fail due to case when one of the cubes is scalar - this fixes that (:pull:`1715`) by :user:`valeriupredoi`
-  Pick up esmvalcore badge instead of esmvaltool one in README (:pull:`1749`) by :user:`valeriupredoi`
-  Restore support for scalar cubes to time selection preprocessor functions (:pull:`1750`) by :user:`bouweandela`
-  Fix calculation of precipitation flux in EMAC on-the-fly CMORizer (:pull:`1755`) by :user:`schlunma`

Deprecations
~~~~~~~~~~~~

-  Remove deprecation warning for regrid schemes already deprecated for v2.7.0 (:pull:`1753`) by :user:`valeriupredoi`

Documentation
~~~~~~~~~~~~~

-  Add Met Office Installation Method (:pull:`1692`) by :user:`mo-tgeddes`
-  Add MO-paths to config file (:pull:`1709`) by :user:`mo-tgeddes`
-  Update MO obs4MIPs paths in the user configuration file (:pull:`1734`) by :user:`mo-tgeddes`
-  Update `Making a release` section of the documentation (:pull:`1689`) by :user:`sloosvel`
-  Added changelog for v2.7.0 (:pull:`1746`) by :user:`valeriupredoi`
-  update CITATION.cff file with 2.7.0 release info (:pull:`1757`) by :user:`valeriupredoi`

Improvements
~~~~~~~~~~~~

-  New preprocessor function 'rolling_window_statistics' (:pull:`1702`) by :user:`malininae`
-  Remove `pytest_flake8` plugin and use `flake8` instead (:pull:`1722`) by :user:`valeriupredoi`
-  Added CESM2 CMORizer (:pull:`1678`) by :user:`schlunma`
-  Speed up functions that use time dimension (:pull:`1713`) by :user:`bouweandela`
-  Modernize and minimize pylint configuration (:pull:`1726`) by :user:`bouweandela`

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Refactored native model fixes by adding common base class `NativeDatasetFix` (:pull:`1694`) by :user:`schlunma`

Installation
~~~~~~~~~~~~

-  Pin `netCDF4 != 1.6.1` since that seems to throw a flurry of Segmentation Faults (:pull:`1724`) by :user:`valeriupredoi`

Automatic testing
~~~~~~~~~~~~~~~~~

-  Pin `flake8<5.0.0` since Circle CI tests are failing copiously (:pull:`1698`) by :user:`valeriupredoi`
-  Added tests for input/output filenames for ICON and EMAC on-the-fly CMORizer (:pull:`1718`) by :user:`schlunma`
-  Fix failed tests for Python<3.10 resulting from typing (:pull:`1748`) by :user:`schlunma`

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

-  Deprecate the function `esmvalcore.var_name_constraint` (:pull:`1592`) by :user:`schlunma`. This function is scheduled for removal in v2.8.0. Please use :class:`iris.NameConstraint` with the keyword argument `var_name` instead: this is an exact replacement.

Bug fixes
~~~~~~~~~

-  Added `start_year` and `end_year` attributes to derived variables (:pull:`1547`) by :user:`schlunma`
-  Show all results on recipe results webpage (:pull:`1560`) by :user:`bouweandela`
-  Regridding regular grids with similar coordinates  (:pull:`1567`) by :user:`tomaslovato`
-  Fix timerange wildcard search when deriving variables or downloading files (:pull:`1562`) by :user:`sloosvel`
-  Fix `force_derivation` bug (:pull:`1627`) by :user:`sloosvel`
-  Correct `build-and-deploy-on-pypi` action (:pull:`1634`) by :user:`sloosvel`
-  Apply `clip_timerange` to time dependent fx variables (:pull:`1603`) by :user:`sloosvel`
-  Correctly handle requests.exceptions.ConnectTimeout when an ESGF index node is offline (:pull:`1638`) by :user:`bouweandela`

CMOR standard
~~~~~~~~~~~~~

-  Added custom CMOR tables used for EMAC CMORizer (:pull:`1599`) by :user:`schlunma`
-  Extended ICON CMORizer (:pull:`1549`) by :user:`schlunma`
-  Add CMOR check exception for a basin coord named sector (:pull:`1612`) by :user:`dhohn`
-  Custom user-defined location for custom CMOR tables (:pull:`1625`) by :user:`schlunma`

Containerization
~~~~~~~~~~~~~~~~

-  Remove update command in Dockerfile (:pull:`1630`) by :user:`sloosvel`

Community
~~~~~~~~~

-  Add David Hohn to contributors' list (:pull:`1586`) by :user:`valeriupredoi`

Documentation
~~~~~~~~~~~~~

-  [Github Actions Docs] Full explanation on how to use the GA test triggered by PR comment and added docs link for GA hosted runners  (:pull:`1553`) by :user:`valeriupredoi`
-  Update the command for building the documentation (:pull:`1556`) by :user:`bouweandela`
-  Update documentation on running the tool (:pull:`1400`) by :user:`bouweandela`
-  Add support for DKRZ-Levante (:pull:`1558`) by :user:`remi-kazeroni`
-  Improved documentation on native dataset support (:pull:`1559`) by :user:`schlunma`
-  Tweak `extract_point` preprocessor: explain what it returns if one point coord outside cube and add explicit test  (:pull:`1584`) by :user:`valeriupredoi`
-  Update CircleCI, readthedocs, and Docker configuration (:pull:`1588`) by :user:`bouweandela`
-  Remove support for Mistral in `config-user.yml` (:pull:`1620`) by :user:`remi-kazeroni`
-  Add changelog for v2.6.0rc1 (:pull:`1633`) by :user:`sloosvel`
-  Add a note on transferring permissions to the release manager (:pull:`1645`) by :user:`bouweandela`
-  Add documentation on building and uploading Docker images (:pull:`1644`) by :user:`bouweandela`
-  Update documentation on ESMValTool module at DKRZ (:pull:`1647`) by :user:`remi-kazeroni`
-  Expanded information on deprecations in changelog (:pull:`1658`) by :user:`schlunma`

Improvements
~~~~~~~~~~~~

-  Removed trailing whitespace in custom CMOR tables (:pull:`1564`) by :user:`schlunma`
-  Try searching multiple ESGF index nodes (:pull:`1561`) by :user:`bouweandela`
-  Add CMIP6 `amoc` derivation case and add a test (:pull:`1577`) by :user:`valeriupredoi`
-  Added EMAC CMORizer (:pull:`1554`) by :user:`schlunma`
-  Improve performance of `volume_statistics` (:pull:`1545`) by :user:`sloosvel`

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Fixes of ocean variables in multiple CMIP6 datasets (:pull:`1566`) by :user:`tomaslovato`
-  Ensure lat/lon bounds in FGOALS-l3 atmos variables are contiguous (:pull:`1571`) by :user:`sloosvel`
-  Added `AllVars` fix for CMIP6's ICON-ESM-LR (:pull:`1582`) by :user:`schlunma`

Installation
~~~~~~~~~~~~

-  Removed `package/meta.yml` (:pull:`1540`) by :user:`schlunma`
-  Pinned iris>=3.2.1 (:pull:`1552`) by :user:`schlunma`
-  Use setuptools-scm to automatically generate the version number (:pull:`1578`) by :user:`bouweandela`
-  Pin cf-units to lower than 3.1.0 to temporarily avoid changes within new version related to calendars (:pull:`1659`) by :user:`valeriupredoi`

Preprocessor
~~~~~~~~~~~~

-  Allowed special case for unit conversion of precipitation (`kg m-2 s-1` <--> `mm day-1`) (:pull:`1574`) by :user:`schlunma`
-  Add general `extract_coordinate_points` preprocessor (:pull:`1581`) by :user:`sloosvel`
-  Add preprocessor `accumulate_coordinate` (:pull:`1281`) by :user:`jvegreg`
-  Add `axis_statistics` and improve `depth_integration` (:pull:`1589`) by :user:`sloosvel`

Release
~~~~~~~

-  Increase version number for ESMValCore v2.6.0rc1 (:pull:`1632`) by :user:`sloosvel`
-  Update changelog and version for 2.6rc3 (:pull:`1646`) by :user:`sloosvel`
-  Add changelog for rc4 (:pull:`1662`) by :user:`sloosvel`


Automatic testing
~~~~~~~~~~~~~~~~~

-  Refresh CircleCI cache weekly (:pull:`1597`) by :user:`bouweandela`
-  Use correct cache restore key on CircleCI (:pull:`1598`) by :user:`bouweandela`
-  Install git and ssh before checking out code on CircleCI (:pull:`1601`) by :user:`bouweandela`
-  Fetch all history in Github Action tests (:pull:`1622`) by :user:`sloosvel`
-  Test Github Actions dashboard badge from meercode.io (:pull:`1640`) by :user:`valeriupredoi`
-  Improve esmvalcore.esgf unit test (:pull:`1650`) by :user:`bouweandela`

Variable Derivation
~~~~~~~~~~~~~~~~~~~

-  Added derivation of `hfns` (:pull:`1594`) by :user:`schlunma`

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

-  Update Cordex section in  `config-developer.yml` (:pull:`1303`) by :user:`francesco-cmcc`. This changes the naming convention of ESMValCore's output files from CORDEX dataset. This only affects recipes that use CORDEX data. Most likely, no changes in diagnostics are necessary; however, if code relies on the specific naming convention of files, it might need to be adapted.
-  Dropped Python 3.7 (:pull:`1530`) by :user:`schlunma`. ESMValCore v2.5.0 dropped support for Python 3.7. From now on Python >=3.8 is required to install ESMValCore. The main reason for this is that conda-forge dropped support for Python 3.7 for OSX and arm64 (more details are given `here <https://github.com/ESMValGroup/ESMValTool/issues/2584#issuecomment-1063853630>`__).

Bug fixes
~~~~~~~~~

-  Fix `extract_shape` when fx vars are present (:pull:`1403`) by :user:`sloosvel`
-  Added support of `extra_facets` to fx variables added by the preprocessor (:pull:`1399`) by :user:`schlunma`
-  Augmented input for derived variables with extra_facets (:pull:`1412`) by :user:`schlunma`
-  Correctly use masked arrays after `unstructured_nearest` regridding (:pull:`1414`) by :user:`schlunma`
-  Fixing the broken derivation script for XCH4 (and XCO2) (:pull:`1428`) by :user:`hb326`
-  Ignore `.pymon-journal` file in test discovery (:pull:`1436`) by :user:`valeriupredoi`
-  Fixed bug that caused automatic download to fail in rare cases (:pull:`1442`) by :user:`schlunma`
-  Add new `JULIA_LOAD_PATH` to diagnostic task test (:pull:`1444`) by :user:`valeriupredoi`
-  Fix provenance file permissions (:pull:`1468`) by :user:`bouweandela`
-  Fixed usage of `statistics=std_dev` option in multi-model statistics preprocessors (:pull:`1478`) by :user:`schlunma`
-  Removed scalar coordinates `p0` and `ptop` prior to merge in `multi_model_statistics` (:pull:`1471`) by :user:`axel-lauer`
-  Added `dataset` and `alias` attributes to `multi_model_statistics` output (:pull:`1483`) by :user:`schlunma`
-  Fixed issues with multi-model-statistics timeranges (:pull:`1486`) by :user:`schlunma`
-  Fixed output messages for CMOR logging (:pull:`1494`) by :user:`schlunma`
-  Fixed `clip_timerange` if only a single time point is extracted (:pull:`1497`) by :user:`schlunma`
-  Fixed chunking in `multi_model_statistics` (:pull:`1500`) by :user:`schlunma`
-  Fixed renaming of auxiliary coordinates in `multi_model_statistics` if coordinates are equal (:pull:`1502`) by :user:`schlunma`
-  Fixed timerange selection for automatic downloads (:pull:`1517`) by :user:`schlunma`
-  Fixed chunking in `multi_model_statistics` (:pull:`1524`) by :user:`schlunma`

Deprecations
~~~~~~~~~~~~

-  Renamed vertical regridding schemes (:pull:`1429`) by :user:`schlunma`. Old regridding schemes are supported until v2.7.0. For details, see :ref:`Vertical interpolation schemes <Vertical interpolation schemes>`.

Documentation
~~~~~~~~~~~~~

-  Remove duplicate entries in changelog (:pull:`1391`) by :user:`zklaus`
-  Documentation on how to use HPC central installations (:pull:`1409`) by :user:`valeriupredoi`
-  Correct brackets in preprocessor documentation for list of seasons (:pull:`1420`) by :user:`bouweandela`
-  Add Python=3.10 to package info, update Circle CI auto install and documentation for Python=3.10 (:pull:`1432`) by :user:`valeriupredoi`
-  Reverted unintentional change in `.zenodo.json` (:pull:`1452`) by :user:`schlunma`
-  Synchronized config-user.yml with version from ESMValTool (:pull:`1453`) by :user:`schlunma`
-  Solved issues in configuration files (:pull:`1457`) by :user:`schlunma`
-  Add direct link to download conda lock file in the install documentation (:pull:`1462`) by :user:`valeriupredoi`
-  CITATION.cff fix and automatic validation of citation metadata (:pull:`1467`) by :user:`valeriupredoi`
-  Updated documentation on how to deprecate features (:pull:`1426`) by :user:`schlunma`
-  Added reference hook to conda lock in documentation install section (:pull:`1473`) by :user:`valeriupredoi`
-  Increased ESMValCore version to 2.5.0rc1 (:pull:`1477`) by :user:`schlunma`
-  Added changelog for v2.5.0 release (:pull:`1476`) by :user:`schlunma`
-  Increased ESMValCore version to 2.5.0rc2 (:pull:`1487`) by :user:`schlunma`
-  Added some authors to citation and zenodo files (:pull:`1488`) by :user:`SarahAlidoost`
-  Restored `scipy` intersphinx mapping (:pull:`1491`) by :user:`schlunma`
-  Increased ESMValCore version to 2.5.0rc3 (:pull:`1504`) by :user:`schlunma`
-  Fix download instructions for the MSWEP dataset (:pull:`1506`) by :user:`remi-kazeroni`
-  Documentation updated for the new cmorizer framework (:pull:`1417`) by :user:`remi-kazeroni`
-  Added tests for duplicates in changelog and removed duplicates (:pull:`1508`) by :user:`schlunma`
-  Increased ESMValCore version to 2.5.0rc4 (:pull:`1519`) by :user:`schlunma`
-  Add Github Actions Test badge in README (:pull:`1526`) by :user:`valeriupredoi`
-  Increased ESMValCore version to 2.5.0rc5 (:pull:`1529`) by :user:`schlunma`
-  Increased ESMValCore version to 2.5.0rc6 (:pull:`1532`) by :user:`schlunma`

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Added fix for AIRS v2.1 (obs4mips) (:pull:`1472`) by :user:`axel-lauer`

Preprocessor
~~~~~~~~~~~~

-  Added bias preprocessor (:pull:`1406`) by :user:`schlunma`
-  Improve error messages when a preprocessor is failing (:pull:`1408`) by :user:`schlunma`
-  Added option to explicitly not use fx variables in preprocessors (:pull:`1416`) by :user:`schlunma`
-  Add `extract_location` preprocessor to extract town, city, mountains etc - anything specifiable by a location (:pull:`1251`) by :user:`jvegreg`
-  Add ensemble statistics preprocessor and 'groupby' option for multimodel (:pull:`673`) by :user:`sloosvel`
-  Generic regridding preprocessor (:pull:`1448`) by :user:`zklaus`

Automatic testing
~~~~~~~~~~~~~~~~~

-  Add `pandas` as dependency :panda_face:  (:pull:`1402`) by :user:`valeriupredoi`
-  Fixed tests for python 3.7 (:pull:`1410`) by :user:`schlunma`
-  Remove accessing `.xml()` cube method from test (:pull:`1419`) by :user:`valeriupredoi`
-  Remove flag to use pip 2020 solver from Github Action pip install command on OSX (:pull:`1357`) by :user:`valeriupredoi`
-  Add Python=3.10 to Github Actions and switch to Python=3.10 for the Github Action that builds the PyPi package (:pull:`1430`) by :user:`valeriupredoi`
-  Pin `flake8<4` to keep getting relevant error traces when tests fail with FLAKE8 issues (:pull:`1434`) by :user:`valeriupredoi`
-  Implementing conda lock (:pull:`1164`) by :user:`valeriupredoi`
-  Relocate `pytest-monitor` outputted database `.pymon` so `.pymon-journal` file should not be looked for by `pytest` (:pull:`1441`) by :user:`valeriupredoi`
-  Switch to Mambaforge in Github Actions tests (:pull:`1438`) by :user:`valeriupredoi`
-  Turn off conda lock file creation on any push on `main` branch from Github Action test (:pull:`1489`) by :user:`valeriupredoi`
-  Add DRS path test for IPSLCM files (:pull:`1490`) by :user:`senesis`
-  Add a test module that runs tests of `iris` I/O every time we notice serious bugs there (:pull:`1510`) by :user:`valeriupredoi`
-  [Github Actions] Trigger Github Actions tests (`run-tests.yml` workflow) from a comment in a PR (:pull:`1520`) by :user:`valeriupredoi`
-  Update Linux condalock file (various pull requests) github-actions[bot]

Installation
~~~~~~~~~~~~

-  Move `nested-lookup` dependency to `environment.yml` to be installed from conda-forge instead of PyPi (:pull:`1481`) by :user:`valeriupredoi`
-  Pinned `iris` (:pull:`1511`) by :user:`schlunma`
-  Updated dependencies (:pull:`1521`) by :user:`schlunma`
-  Pinned iris<3.2.0 (:pull:`1525`) by :user:`schlunma`

Improvements
~~~~~~~~~~~~

-  Allow to load all files, first X years or last X years in an experiment (:pull:`1133`) by :user:`sloosvel`
-  Filter tasks earlier (:pull:`1264`) by :user:`jvegreg`
-  Added earlier validation for command line arguments (:pull:`1435`) by :user:`schlunma`
-  Remove `profile_diagnostic` from diagnostic settings and increase test coverage of `_task.py` (:pull:`1404`) by :user:`valeriupredoi`
-  Add `output2` to the `product` extra facet of CMIP5 data (:pull:`1514`) by :user:`remi-kazeroni`
-  Speed up ESGF search (:pull:`1512`) by :user:`bouweandela`


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

-  Crop on the ID-selected region(s) and not on the whole shapefile (:pull:`1151`) by :user:`stefsmeets`
-  Add 'comment' to list of removed attributes (:pull:`1244`) by :user:`Peter9192`
-  Speed up multimodel statistics and fix bug in peak computation (:pull:`1301`) by :user:`bouweandela`
-  No longer make plots of provenance (:pull:`1307`) by :user:`bouweandela`
-  No longer embed provenance in output files (:pull:`1306`) by :user:`bouweandela`
-  Removed automatic addition of areacello to obs4mips datasets (:pull:`1316`) by :user:`schlunma`
-  Pin docutils <0.17 to fix bullet lists on readthedocs (:pull:`1320`) by :user:`zklaus`
-  Fix obs4MIPs capitalization (:pull:`1328`) by :user:`bouweandela`
-  Fix Python 3.7 tests (:pull:`1330`) by :user:`bouweandela`
-  Handle fx variables in `extract_levels` and some time operations (:pull:`1269`) by :user:`sloosvel`
-  Refactored mask regridding for irregular grids (fixes #772) (:pull:`865`) by :user:`zklaus`
-  Fix `da.broadcast_to` call when the fx cube has different shape than target data cube (:pull:`1350`) by :user:`valeriupredoi`
-  Add tests for _aggregate_time_fx (:pull:`1354`) by :user:`sloosvel`
-  Fix extra facets (:pull:`1360`) by :user:`bouweandela`
-  Pin pip!=21.3 to avoid pypa/pip#10573 with editable installs (:pull:`1359`) by :user:`zklaus`
-  Add a custom `date2num` function to deal with changes in cftime (:pull:`1373`) by :user:`zklaus`
-  Removed custom version of `AtmosphereSigmaFactory` (:pull:`1382`) by :user:`schlunma`

Deprecations
~~~~~~~~~~~~

-  Remove write_netcdf and write_plots from config-user.yml (:pull:`1300`) by :user:`bouweandela`

Documentation
~~~~~~~~~~~~~

-  Add link to plot directory in index.html (:pull:`1256`) by :user:`stefsmeets`
-  Work around issue with yapf not following PEP8 (:pull:`1277`) by :user:`bouweandela`
-  Update the core development team (:pull:`1278`) by :user:`bouweandela`
-  Update the documentation of the provenance interface (:pull:`1305`) by :user:`bouweandela`
-  Update version number to first release candidate 2.4.0rc1 (:pull:`1363`) by :user:`zklaus`
-  Update to new ESMValTool logo (:pull:`1374`) by :user:`zklaus`
-  Update version number for third release candidate 2.4.0rc3 (:pull:`1384`) by :user:`zklaus`
-  Update changelog for 2.4.0rc3 (:pull:`1385`) by :user:`zklaus`
-  Update version number to final 2.4.0 release (:pull:`1389`) by :user:`zklaus`
-  Update changelog for 2.4.0 (:pull:`1366`) by :user:`zklaus`

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Add fix for differing latitude coordinate between historical and ssp585 in MPI-ESM1-2-HR r2i1p1f1 (:pull:`1292`) by :user:`bouweandela`
-  Add fixes for time and latitude coordinate of EC-Earth3 r3i1p1f1 (:pull:`1290`) by :user:`bouweandela`
-  Apply latitude fix to all CCSM4 variables (:pull:`1295`) by :user:`bouweandela`
-  Fix lat and lon bounds for FGOALS-g3 mrsos (:pull:`1289`) by :user:`thomascrocker`
-  Add grid fix for tos in fgoals-f3-l (:pull:`1326`) by :user:`sloosvel`
-  Add fix for CIESM pr (:pull:`1344`) by :user:`bouweandela`
-  Fix DRS for IPSLCM : split attribute 'freq' into : 'out' and 'freq' (:pull:`1304`) by :user:`senesis`

CMOR standard
~~~~~~~~~~~~~

-  Remove history attribute from coords (:pull:`1276`) by :user:`jvegreg`
-  Increased flexibility of CMOR checks for datasets with generic alevel coordinates (:pull:`1032`) by :user:`schlunma`
-  Automatically fix small deviations in vertical levels (:pull:`1177`) by :user:`bouweandela`
-  Adding standard names to the custom tables of the `rlns` and `rsns` variables (:pull:`1386`) by :user:`remi-kazeroni`

Preprocessor
~~~~~~~~~~~~

-  Implemented fully lazy climate_statistics (:pull:`1194`) by :user:`schlunma`
-  Run the multimodel statistics preprocessor last (:pull:`1299`) by :user:`bouweandela`

Automatic testing
~~~~~~~~~~~~~~~~~

-  Improving test coverage for _task.py (:pull:`514`) by :user:`valeriupredoi`
-  Upload coverage to codecov (:pull:`1190`) by :user:`bouweandela`
-  Improve codecov status checks (:pull:`1195`) by :user:`bouweandela`
-  Fix curl install in CircleCI (:pull:`1228`) by :user:`jvegreg`
-  Drop support for Python 3.6 (:pull:`1200`) by :user:`valeriupredoi`
-  Allow more recent version of `scipy` (:pull:`1182`) by :user:`schlunma`
-  Speed up conda build `conda_build` Circle test by using `mamba` solver via `boa` (and use it for Github Actions test too) (:pull:`1243`) by :user:`valeriupredoi`
-  Fix numpy deprecation warnings (:pull:`1274`) by :user:`bouweandela`
-  Unpin upper bound for iris (previously was at <3.0.4)  (:pull:`1275`) by :user:`valeriupredoi`
-  Modernize `conda_install` test on Circle CI by installing from conda-forge with Python 3.9 and change install instructions in documentation (:pull:`1280`) by :user:`valeriupredoi`
-  Run a nightly Github Actions workflow to monitor tests memory per test (configurable for other metrics too) (:pull:`1284`) by :user:`valeriupredoi`
-  Speed up tests of tasks (:pull:`1302`) by :user:`bouweandela`
-  Fix upper case to lower case variables and functions for flake compliance in `tests/unit/preprocessor/_regrid/test_extract_levels.py` (:pull:`1347`) by :user:`valeriupredoi`
-  Cleaned up a bit Github Actions workflows (:pull:`1345`) by :user:`valeriupredoi`
-  Update circleci jobs: renaming tests to more descriptive names and removing conda build test (:pull:`1351`) by :user:`zklaus`
-  Pin iris to latest `>=3.1.0` (:pull:`1341`) by :user:`valeriupredoi`

Installation
~~~~~~~~~~~~

-  Pin esmpy to anything but 8.1.0 since that particular one changes the CPU affinity (:pull:`1310`) by :user:`valeriupredoi`

Improvements
~~~~~~~~~~~~

-  Add a more friendly and useful message when using default config file (:pull:`1233`) by :user:`valeriupredoi`
-  Replace os.walk by glob.glob in data finder (only look for data in the specified locations) (:pull:`1261`) by :user:`bouweandela`
-  Machine-specific directories for auxiliary data in the `config-user.yml` file (:pull:`1268`) by :user:`remi-kazeroni`
-  Add an option to download missing data from ESGF (:pull:`1217`) by :user:`bouweandela`
-  Speed up provenance recording (:pull:`1327`) by :user:`bouweandela`
-  Improve results web page (:pull:`1332`) by :user:`bouweandela`
-  Move institutes from config-developer.yml to default extra facets config and add wildcard support for extra facets (:pull:`1259`) by :user:`bouweandela`
-  Add support for re-using preprocessor output from previous runs (:pull:`1321`) by :user:`bouweandela`
-  Log fewer messages to screen and hide stack trace for known recipe errors (:pull:`1296`) by :user:`bouweandela`
-  Log ESMValCore and ESMValTool versions when running (:pull:`1263`) by :user:`jvegreg`
-  Add "grid" as a tag to the output file template for CMIP6 (:pull:`1356`) by :user:`zklaus`
-  Implemented ICON project to read native ICON model output (:pull:`1079`) by :user:`bsolino`


.. _changelog-v2-3-1:

v2.3.1
------

This release includes

Bug fixes
~~~~~~~~~

-  Update config-user.yml template with correct drs entries for CEDA-JASMIN (:pull:`1184`) by :user:`valeriupredoi`
-  Enhancing MIROC5 fix for hfls and evspsbl (:pull:`1192`) by :user:`katjaweigel`
-  Fix alignment of daily data with inconsistent calendars in multimodel statistics (:pull:`1212`) by :user:`Peter9192`
-  Pin cf-units, remove github actions test for Python 3.6 and fix test_access1_0 and test_access1_3 to use cf-units for comparisons (:pull:`1197`) by :user:`valeriupredoi`
-  Fixed search for fx files when no ``mip`` is given (:pull:`1216`) by :user:`schlunma`
-  Make sure climate statistics always returns original dtype (:pull:`1237`) by :user:`zklaus`
-  Bugfix for regional regridding when non-integer range is passed (:pull:`1231`) by :user:`stefsmeets`
-  Make sure area_statistics preprocessor always returns original dtype (:pull:`1239`) by :user:`zklaus`
-  Add "." (dot) as allowed separation character for the time range group (:pull:`1248`) by :user:`zklaus`

Documentation
~~~~~~~~~~~~~

-  Add a link to the instructions to use pre-installed versions on HPC clusters (:pull:`1186`) by :user:`remi-kazeroni`
-  Bugfix release: set version to 2.3.1 (:pull:`1253`) by :user:`zklaus`

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Set circular attribute in MCM-UA-1-0 fix (:pull:`1178`) by :user:`sloosvel`
-  Fixed time coordinate of MIROC-ESM (:pull:`1188`) by :user:`schlunma`

Preprocessor
~~~~~~~~~~~~

-  Filter warnings about collapsing multi-model dimension in multimodel statistics preprocessor function (:pull:`1215`) by :user:`bouweandela`
-  Remove fx variables before computing multimodel statistics (:pull:`1220`) by :user:`sloosvel`

Installation
~~~~~~~~~~~~

-  Pin lower bound for iris to 3.0.2 (:pull:`1206`) by :user:`valeriupredoi`
-  Pin `iris<3.0.4` to ensure we still (sort of) support Python 3.6 (:pull:`1252`) by :user:`valeriupredoi`

Improvements
~~~~~~~~~~~~

-  Add test to verify behaviour for scalar height coord for tas in multi-model (:pull:`1209`) by :user:`Peter9192`
-  Sort missing years in "No input data available for years" message (:pull:`1225`) by :user:`ledm`


.. _changelog-v2-3-0:

v2.3.0
------

This release includes

Bug fixes
~~~~~~~~~

-  Extend preprocessor multi_model_statistics to handle data with "altitude" coordinate (:pull:`1010`) by :user:`axel-lauer`
-  Remove scripts included with CMOR tables (:pull:`1011`) by :user:`bouweandela`
-  Avoid side effects in extract_season (:pull:`1019`) by :user:`jvegreg`
-  Use nearest scheme to avoid interpolation errors with masked data in regression test (:pull:`1021`) by :user:`stefsmeets`
-  Move _get_time_bounds from preprocessor._time to cmor.check to avoid circular import with cmor module (:pull:`1037`) by :user:`valeriupredoi`
-  Fix test that makes conda build fail (:pull:`1046`) by :user:`valeriupredoi`
-  Fix 'positive' attribute for rsns/rlns variables (:pull:`1051`) by :user:`lukasbrunner`
-  Added preprocessor mask_multimodel (:pull:`767`) by :user:`schlunma`
-  Fix bug when fixing bounds after fixing longitude values (:pull:`1057`) by :user:`sloosvel`
-  Run conda build parallel AND sequential tests (:pull:`1065`) by :user:`valeriupredoi`
-  Add key to id_prop (:pull:`1071`) by :user:`lukasbrunner`
-  Fix bounds after reversing coordinate values (:pull:`1061`) by :user:`sloosvel`
-  Fixed --skip-nonexistent option (:pull:`1093`) by :user:`schlunma`
-  Do not consider CMIP5 variable sit to be the same as sithick from CMIP6 (:pull:`1033`) by :user:`bouweandela`
-  Improve finding date range in filenames (enforces separators) (:pull:`1145`) by :user:`senesis`
-  Review fx handling (:pull:`1147`) by :user:`sloosvel`
-  Fix lru cache decorator with explicit call to method (:pull:`1172`) by :user:`valeriupredoi`
-  Update _volume.py (:pull:`1174`) by :user:`ledm`

Deprecations
~~~~~~~~~~~~



Documentation
~~~~~~~~~~~~~

-  Final changelog for 2.3.0 (:pull:`1163`) by :user:`zklaus`
-  Set version to 2.3.0 (:pull:`1162`) by :user:`zklaus`
-  Fix documentation build (:pull:`1006`) by :user:`bouweandela`
-  Add labels required for linking from ESMValTool docs (:pull:`1038`) by :user:`bouweandela`
-  Update contribution guidelines (:pull:`1047`) by :user:`bouweandela`
-  Fix basestring references in documentation (:pull:`1106`) by :user:`jvegreg`
-  Updated references master to main (:pull:`1132`) by :user:`axel-lauer`
-  Add instructions how to use the central installation at DKRZ-Mistral (:pull:`1155`) by :user:`remi-kazeroni`

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Added fixes for various CMIP5 datasets, variable cl (3-dim cloud fraction) (:pull:`1017`) by :user:`axel-lauer`
-  Added fixes for hybrid level coordinates of CESM2 models (:pull:`882`) by :user:`schlunma`
-  Extending LWP fix for CMIP6 models (:pull:`1049`) by :user:`axel-lauer`
-  Add fixes for the net & up radiation variables from ERA5 (:pull:`1052`) by :user:`lukasbrunner`
-  Add derived variable rsus (:pull:`1053`) by :user:`lukasbrunner`
-  Supported `mip`-level fixes (:pull:`1095`) by :user:`schlunma`
-  Fix erroneous use of `grid_latitude` and `grid_longitude` and cleaned ocean grid fixes (:pull:`1092`) by :user:`schlunma`
-  Fix for pr of miroc5 (:pull:`1110`) by :user:`remi-kazeroni`
-  Ocean depth fix for cnrm_esm2_1, gfdl_esm4, ipsl_cm6a_lr datasets +  mcm_ua_1_0 (:pull:`1098`) by :user:`tomaslovato`
-  Fix for uas variable of the MCM_UA_1_0 dataset (:pull:`1102`) by :user:`remi-kazeroni`
-  Fixes for sos and siconc of BCC models (:pull:`1090`) by :user:`remi-kazeroni`
-  Run fgco2 fix for all CESM2 models (:pull:`1108`) by :user:`LisaBock`
-  Fixes for the siconc variable of CMIP6 models (:pull:`1105`) by :user:`remi-kazeroni`
-  Fix wrong sign for land surface flux (:pull:`1113`) by :user:`LisaBock`
-  Fix for pr of EC_EARTH (:pull:`1116`) by :user:`remi-kazeroni`

CMOR standard
~~~~~~~~~~~~~

-  Format cmor related files (:pull:`976`) by :user:`jvegreg`
-  Check presence of time bounds and guess them if needed (:pull:`849`) by :user:`sloosvel`
-  Add custom variable "tasaga" (:pull:`1118`) by :user:`LisaBock`
-  Find files for CMIP6 DCPP startdates (:pull:`771`) by :user:`sloosvel`

Preprocessor
~~~~~~~~~~~~

-  Update tests for multimodel statistics preprocessor (:pull:`1023`) by :user:`stefsmeets`
-  Raise in extract_season and extract_month if result is None (:pull:`1041`) by :user:`jvegreg`
-  Allow selection of shapes in extract_shape (:pull:`764`) by :user:`jvegreg`
-  Add option for regional regridding to regrid preprocessor (:pull:`1034`) by :user:`stefsmeets`
-  Load fx variables as cube cell measures / ancillary variables (:pull:`999`) by :user:`sloosvel`
-  Check horizontal grid before regridding (:pull:`507`) by :user:`BenMGeo`
-  Clip irregular grids (:pull:`245`) by :user:`bouweandela`
-  Use native iris functions in multi-model statistics (:pull:`1150`) by :user:`Peter9192`

Notebook API (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~



Automatic testing
~~~~~~~~~~~~~~~~~

-  Report coverage for tests that run on any pull request (:pull:`994`) by :user:`bouweandela`
-  Install ESMValTool sample data from PyPI (:pull:`998`) by :user:`jvegreg`
-  Fix tests for multi-processing with spawn method (i.e. macOSX with Python>3.8) (:pull:`1003`) by :user:`bvreede`
-  Switch to running the Github Action test workflow every 3 hours in single thread mode to observe if Segmentation Faults occur (:pull:`1022`) by :user:`valeriupredoi`
-  Revert to original Github Actions test workflow removing the 3-hourly test run with -n 1 (:pull:`1025`) by :user:`valeriupredoi`
-  Avoid stale cache for multimodel statistics regression tests (:pull:`1030`) by :user:`bouweandela`
-  Add newer Python versions in OSX to Github Actions (:pull:`1035`) by :user:`bvreede`
-  Add tests for type annotations with mypy (:pull:`1042`) by :user:`stefsmeets`
-  Run problematic cmor tests sequentially to avoid segmentation faults on CircleCI (:pull:`1064`) by :user:`valeriupredoi`
-  Test installation of esmvalcore from conda-forge (:pull:`1075`) by :user:`valeriupredoi`
-  Added additional test cases for integration tests of data_finder.py (:pull:`1087`) by :user:`schlunma`
-  Pin cf-units and fix tests (cf-units>=2.1.5) (:pull:`1140`) by :user:`valeriupredoi`
-  Fix failing CircleCI tests (:pull:`1167`) by :user:`bouweandela`
-  Fix test failing due to fx files chosen differently on different OS's (:pull:`1169`) by :user:`valeriupredoi`
-  Compare datetimes instead of strings in _fixes/cmip5/test_access1_X.py (:pull:`1173`) by :user:`valeriupredoi`
-  Pin Python to 3.9 in environment.yml on CircleCI and skip mypy tests in conda build (:pull:`1176`) by :user:`bouweandela`

Installation
~~~~~~~~~~~~

-  Update yamale to version 3 (:pull:`1059`) by :user:`zklaus`

Improvements
~~~~~~~~~~~~

-  Refactor diagnostics / tags management (:pull:`939`) by :user:`stefsmeets`
-  Support multiple paths in input_dir (:pull:`1000`) by :user:`jvegreg`
-  Generate HTML report with recipe output (:pull:`991`) by :user:`stefsmeets`
-  Add timeout to requests.get in _citation.py (:pull:`1091`) by :user:`SarahAlidoost`
-  Add SYNDA drs for CMIP5 and CMIP6 (closes #582) (:pull:`583`) by :user:`zklaus`
-  Add basic support for variable mappings (:pull:`1124`) by :user:`zklaus`
-  Handle IPSL-CM6  (:pull:`1153`) by :user:`senesis`


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

-  Fix path settings for DKRZ/Mistral (:pull:`852`) by :user:`bouweandela`
-  Change logic for calling the diagnostic script to avoid problems with scripts where the executable bit is accidentally set (:pull:`877`) by :user:`bouweandela`
-  Fix overwriting in generic level check (:pull:`886`) by :user:`sloosvel`
-  Add double quotes to script args in rerun screen message when using vprof profiling (:pull:`897`) by :user:`valeriupredoi`
-  Simplify time handling in multi-model statistics preprocessor (:pull:`685`) by :user:`Peter9192`
-  Fix links to Iris documentation (:pull:`966`) by :user:`jvegreg`
-  Bugfix: Fix units for MSWEP data (:pull:`986`) by :user:`stefsmeets`

Deprecations
~~~~~~~~~~~~

-  Deprecate defining write_plots and write_netcdf in config-user file (:pull:`808`) by :user:`bouweandela`

Documentation
~~~~~~~~~~~~~

-  Fix numbering of steps in release instructions (:pull:`838`) by :user:`bouweandela`
-  Add labels to changelogs of individual versions for easy reference (:pull:`899`) by :user:`zklaus`
-  Make CircleCI badge specific to main branch (:pull:`902`) by :user:`bouweandela`
-  Fix docker build badge url (:pull:`906`) by :user:`stefsmeets`
-  Update github PR template (:pull:`909`) by :user:`stefsmeets`
-  Refer to ESMValTool GitHub discussions page in the error message (:pull:`900`) by :user:`bouweandela`
-  Support automatically closing issues (:pull:`922`) by :user:`bouweandela`
-  Fix checkboxes in PR template (:pull:`931`) by :user:`stefsmeets`
-  Change in config-user defaults and documentation with new location for esmeval OBS data on JASMIN (:pull:`958`) by :user:`valeriupredoi`
-  Update Core Team info (:pull:`942`) by :user:`axel-lauer`
-  Update iris documentation URL for sphinx (:pull:`964`) by :user:`bouweandela`
-  Set version to 2.2.0 (:pull:`977`) by :user:`jvegreg`
-  Add first draft of v2.2.0 changelog (:pull:`983`) by :user:`jvegreg`
-  Add checkbox in PR template to assign labels (:pull:`985`) by :user:`jvegreg`
-  Update install.rst (:pull:`848`) by :user:`bascrezee`
-  Change the order of the publication steps (:pull:`984`) by :user:`jvegreg`
-  Add instructions how to use esmvaltool from HPC central installations (:pull:`841`) by :user:`valeriupredoi`

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Fixing unit for derived variable rsnstcsnorm to prevent overcorrection2 (:pull:`846`) by :user:`katjaweigel`
-  Cmip6 fix awi cm 1 1 mr (:pull:`822`) by :user:`mwjury`
-  Cmip6 fix ec earth3 veg (:pull:`836`) by :user:`mwjury`
-  Changed latitude longitude fix from Tas to AllVars. (:pull:`916`) by :user:`katjaweigel`
-  Fix for precipitation (pr) to use ERA5-Land cmorizer (:pull:`879`) by :user:`katjaweigel`
-  Cmip6 fix ec earth3 (:pull:`837`) by :user:`mwjury`
-  Cmip6_fix_fgoals_f3_l_Amon_time_bnds (:pull:`831`) by :user:`mwjury`
-  Fix for FGOALS-f3-L sftlf (:pull:`667`) by :user:`mwjury`
-  Improve ACCESS-CM2 and ACCESS-ESM1-5 fixes and add CIESM and CESM2-WACCM-FV2 fixes for cl, clw and cli (:pull:`635`) by :user:`axel-lauer`
-  Add  fixes for cl, cli, clw and tas for several CMIP6 models (:pull:`955`) by :user:`schlunma`
-  Dataset fixes for MSWEP (:pull:`969`) by :user:`stefsmeets`
-  Dataset fixes for: ACCESS-ESM1-5, CanESM5, CanESM5 for carbon cycle (:pull:`947`) by :user:`bettina-gier`
-  Fixes for KIOST-ESM (CMIP6) (:pull:`904`) by :user:`remi-kazeroni`
-  Fixes for AWI-ESM-1-1-LR (CMIP6, piControl) (:pull:`911`) by :user:`remi-kazeroni`

CMOR standard
~~~~~~~~~~~~~

-  CMOR check generic level coordinates in CMIP6 (:pull:`598`) by :user:`sloosvel`
-  Update CMIP6 tables to 6.9.33 (:pull:`919`) by :user:`jvegreg`
-  Adding custom variables for tas uncertainty (:pull:`924`) by :user:`LisaBock`
-  Remove monotonicity coordinate check for unstructured grids (:pull:`965`) by :user:`jvegreg`

Preprocessor
~~~~~~~~~~~~

-  Added clip_start_end_year preprocessor (:pull:`796`) by :user:`schlunma`
-  Add support for downloading CMIP6 data with Synda (:pull:`699`) by :user:`bouweandela`
-  Add multimodel tests using real data (:pull:`856`) by :user:`stefsmeets`
-  Add plev/altitude conversion to extract_levels (:pull:`892`) by :user:`axel-lauer`
-  Add possibility of custom season extraction. (:pull:`247`) by :user:`mwjury`
-  Adding the ability to derive xch4  (:pull:`783`) by :user:`hb326`
-  Add preprocessor function to resample time and compute x-hourly statistics (:pull:`696`) by :user:`jvegreg`
-  Fix duplication in preprocessors DEFAULT_ORDER introduced in #696 (:pull:`973`) by :user:`jvegreg`
-  Use consistent precision in multi-model statistics calculation and update reference data for tests (:pull:`941`) by :user:`Peter9192`
-  Refactor multi-model statistics code to facilitate ensemble stats and lazy evaluation (:pull:`949`) by :user:`Peter9192`
-  Add option to exclude input cubes in output of multimodel statistics to solve an issue introduced by #949 (:pull:`978`) by :user:`Peter9192`


Automatic testing
~~~~~~~~~~~~~~~~~

-  Pin cftime>=1.3.0 to have newer string formatting in and fix two tests (:pull:`878`) by :user:`valeriupredoi`
-  Switched miniconda conda setup hooks for Github Actions workflows (:pull:`873`) by :user:`valeriupredoi`
-  Add test for latest version resolver (:pull:`874`) by :user:`stefsmeets`
-  Update codacy coverage reporter to fix coverage (:pull:`905`) by :user:`nielsdrost`
-  Avoid hardcoded year in tests and add improvement to plev test case (:pull:`921`) by :user:`bouweandela`
-  Pin scipy to less than 1.6.0 until :issue:`927` gets resolved (:pull:`928`) by :user:`valeriupredoi`
-  Github Actions: change time when conda install test runs (:pull:`930`) by :user:`valeriupredoi`
-  Remove redundant test line from test_utils.py (:pull:`935`) by :user:`valeriupredoi`
-  Removed netCDF4 package from integration tests of fixes (:pull:`938`) by :user:`schlunma`
-  Use new conda environment for installing ESMValCore in Docker containers (:pull:`951`) by :user:`bouweandela`

Notebook API (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Implement importable config object in experimental API submodule (:pull:`868`) by :user:`stefsmeets`
-  Add loading and running recipes to the notebook API (:pull:`907`) by :user:`stefsmeets`
-  Add displaying and loading of recipe output to the notebook API (:pull:`957`) by :user:`stefsmeets`
-  Add functionality to run single diagnostic task to notebook API (:pull:`962`) by :user:`stefsmeets`

Improvements
~~~~~~~~~~~~

-  Create CODEOWNERS file (:pull:`809`) by :user:`jvegreg`
-  Remove code needed for Python <3.6 (:pull:`844`) by :user:`bouweandela`
-  Add requests as a dependency (:pull:`850`) by :user:`bouweandela`
-  Pin Python to less than 3.9 (:pull:`870`) by :user:`valeriupredoi`
-  Remove numba dependency (:pull:`880`) by :user:`schlunma`
-  Add Listing and finding recipes to the experimental notebook API (:pull:`901`) by :user:`stefsmeets`
-  Skip variables that don't have dataset or additional_dataset keys (:pull:`860`) by :user:`valeriupredoi`
-  Refactor logging configuration (:pull:`933`) by :user:`stefsmeets`
-  Xco2 derivation (:pull:`913`) by :user:`bettina-gier`
-  Working environment for Python 3.9 (pin to !=3.9.0) (:pull:`885`) by :user:`valeriupredoi`
-  Print source file when using config get_config_user command (:pull:`960`) by :user:`valeriupredoi`
-  Switch to Iris 3 (:pull:`819`) by :user:`stefsmeets`
-  Refactor tasks (:pull:`959`) by :user:`stefsmeets`
-  Restore task summary in debug log after #959 (:pull:`981`) by :user:`bouweandela`
-  Pin pre-commit hooks (:pull:`974`) by :user:`stefsmeets`
-  Improve error messages when data is missing (:pull:`917`) by :user:`jvegreg`
-  Set remove_preproc_dir to false in default config-user (:pull:`979`) by :user:`valeriupredoi`
-  Move fiona to be installed from conda forge (:pull:`987`) by :user:`valeriupredoi`
-  Re-added fiona in setup.py (:pull:`990`) by :user:`valeriupredoi`

.. _changelog-v2-1-0:

v2.1.0
------

This release includes

Bug fixes
~~~~~~~~~

-  Set unit=1 if anomalies are standardized (:pull:`727`) by :user:`bascrezee`
-  Fix crash for FGOALS-g2 variables without longitude coordinate (:pull:`729`) by :user:`bouweandela`
-  Improve variable alias management (:pull:`595`) by :user:`jvegreg`
-  Fix area_statistics fx files loading (:pull:`798`) by :user:`jvegreg`
-  Fix units after derivation (:pull:`754`) by :user:`schlunma`

Documentation
~~~~~~~~~~~~~

-  Update v2.0.0 release notes with final additions (:pull:`722`) by :user:`bouweandela`
-  Update package description in setup.py (:pull:`725`) by :user:`mattiarighi`
-  Add installation instructions for pip installation (:pull:`735`) by :user:`bouweandela`
-  Improve config-user documentation (:pull:`740`) by :user:`bouweandela`
-  Update the zenodo file with contributors (:pull:`807`) by :user:`valeriupredoi`
-  Improve command line run documentation (:pull:`721`) by :user:`jvegreg`
-  Update the zenodo file with contributors (continued) (:pull:`810`) by :user:`valeriupredoi`

Improvements
~~~~~~~~~~~~

-  Reduce size of docker image (:pull:`723`) by :user:`jvegreg`
-  Add 'test' extra to installation, used by docker development tag (:pull:`733`) by :user:`bouweandela`
-  Correct dockerhub link (:pull:`736`) by :user:`bouweandela`
-  Create action-install-from-pypi.yml (:pull:`734`) by :user:`valeriupredoi`
-  Add pre-commit for linting/formatting (:pull:`766`) by :user:`stefsmeets`
-  Run tests in parallel and when building conda package (:pull:`745`) by :user:`bouweandela`
-  Readable exclude pattern for pre-commit (:pull:`770`) by :user:`stefsmeets`
-  Github Actions Tests (:pull:`732`) by :user:`valeriupredoi`
-  Remove isort setup to fix formatting conflict with yapf (:pull:`778`) by :user:`stefsmeets`
-  Fix yapf-isort import formatting conflict (Fixes #777) (:pull:`784`) by :user:`stefsmeets`
-  Sorted output for `esmvaltool recipes list` (:pull:`790`) by :user:`stefsmeets`
-  Replace vmprof with vprof (:pull:`780`) by :user:`valeriupredoi`
-  Update CMIP6 tables to 6.9.32 (:pull:`706`) by :user:`jvegreg`
-  Default config-user path now set in config-user read function (:pull:`791`) by :user:`jvegreg`
-  Add custom variable lweGrace (:pull:`692`) by :user:`bascrezee`
- Create Github Actions workflow to build and deploy on Test PyPi and PyPi (:pull:`820`) by :user:`valeriupredoi`
- Build and publish the esmvalcore package to conda via Github Actions workflow (:pull:`825`) by :user:`valeriupredoi`

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Fix cmip6 models (:pull:`629`) by :user:`npgillett`
-  Fix siconca variable in EC-Earth3 and EC-Earth3-Veg models in amip simulation (:pull:`702`) by :user:`egalytska`

Preprocessor
~~~~~~~~~~~~

-  Move cmor_check_data to early in preprocessing chain (:pull:`743`) by :user:`bouweandela`
-  Add RMS iris analysis operator to statistics preprocessor functions (:pull:`747`) by :user:`pcosbsc`
-  Add surface chlorophyll concentration as a derived variable (:pull:`720`) by :user:`sloosvel`
-  Use dask to reduce memory consumption of extract_levels for masked data (:pull:`776`) by :user:`valeriupredoi`

.. _changelog-v2-0-0:

v2.0.0
------

This release includes

Bug fixes
~~~~~~~~~

-  Fixed derivation of co2s (:pull:`594`) by :user:`schlunma`
-  Padding while cropping needs to stay within sane bounds for shapefiles that span the whole Earth (:pull:`626`) by :user:`valeriupredoi`
-  Fix concatenation of a single cube (:pull:`655`) by :user:`bouweandela`
-  Fix mask fx dict handling not to fail if empty list in values (:pull:`661`) by :user:`valeriupredoi`
-  Preserve metadata during anomalies computation when using iris cubes difference (:pull:`652`) by :user:`valeriupredoi`
-  Avoid crashing when there is directory 'esmvaltool' in the current working directory (:pull:`672`) by :user:`valeriupredoi`
-  Solve bug in ACCESS1 dataset fix for calendar.  (:pull:`671`) by :user:`Peter9192`
-  Fix the syntax for adding multiple ensemble members from the same dataset (:pull:`678`) by :user:`SarahAlidoost`
-  Fix bug that made preprocessor with fx files fail in rare cases (:pull:`670`) by :user:`schlunma`
-  Add support for string coordinates (:pull:`657`) by :user:`jvegreg`
-  Fixed the shape extraction to account for wraparound shapefile coords (:pull:`319`) by :user:`valeriupredoi`
-  Fixed bug in time weights calculation (:pull:`695`) by :user:`schlunma`
-  Fix diagnostic filter (:pull:`713`) by :user:`jvegreg`

Documentation
~~~~~~~~~~~~~

-  Add pandas as a requirement for building the documentation (:pull:`607`) by :user:`bouweandela`
-  Document default order in which preprocessor functions are applied (:pull:`633`) by :user:`bouweandela`
-  Add pointers about data loading and CF standards to documentation (:pull:`571`) by :user:`valeriupredoi`
-  Config file populated with site-specific data paths examples (:pull:`619`) by :user:`valeriupredoi`
-  Update Codacy badges (:pull:`643`) by :user:`bouweandela`
-  Update copyright info on readthedocs (:pull:`668`) by :user:`bouweandela`
-  Updated references to documentation (now docs.esmvaltool.org) (:pull:`675`) by :user:`axel-lauer`
-  Add all European grants to Zenodo (:pull:`680`) by :user:`bouweandela`
-  Update Sphinx to v3 or later (:pull:`683`) by :user:`bouweandela`
-  Increase version to 2.0.0 and add release notes (:pull:`691`) by :user:`bouweandela`
-  Update setup.py and README.md for use on PyPI (:pull:`693`) by :user:`bouweandela`
-  Suggested Documentation changes (:pull:`690`) by :user:`ssmithClimate`

Improvements
~~~~~~~~~~~~

-  Reduce the size of conda package (:pull:`606`) by :user:`bouweandela`
-  Add a few unit tests for DiagnosticTask (:pull:`613`) by :user:`bouweandela`
-  Make ncl or R tests not fail if package not installed (:pull:`610`) by :user:`valeriupredoi`
-  Pin flake8<3.8.0 (:pull:`623`) by :user:`valeriupredoi`
-  Log warnings for likely errors in provenance record (:pull:`592`) by :user:`bouweandela`
-  Unpin flake8 (:pull:`646`) by :user:`bouweandela`
-  More flexible native6 default DRS (:pull:`645`) by :user:`bouweandela`
-  Try to use the same python for running diagnostics as for esmvaltool (:pull:`656`) by :user:`bouweandela`
-  Fix test for lower python version and add note on lxml (:pull:`659`) by :user:`valeriupredoi`
-  Added 1m deep average soil moisture variable (:pull:`664`) by :user:`bascrezee`
-  Update docker recipe (:pull:`603`) by :user:`jvegreg`
-  Improve command line interface (:pull:`605`) by :user:`jvegreg`
-  Remove utils directory (:pull:`697`) by :user:`bouweandela`
-  Avoid pytest version that crashes (:pull:`707`) by :user:`bouweandela`
-  Options arg in read_config_user_file now optional (:pull:`716`) by :user:`jvegreg`
-  Produce a readable warning if ancestors are a string instead of a list. (:pull:`711`) by :user:`katjaweigel`
-  Pin Yamale to v2 (:pull:`718`) by :user:`bouweandela`
-  Expanded cmor public API (:pull:`714`) by :user:`schlunma`

Fixes for datasets
~~~~~~~~~~~~~~~~~~

-  Added various fixes for hybrid height coordinates (:pull:`562`) by :user:`schlunma`
-  Extended fix for cl-like variables of CESM2 models (:pull:`604`) by :user:`schlunma`
-  Added fix to convert "geopotential" to "geopotential height" for ERA5 (:pull:`640`) by :user:`egalytska`
-  Do not fix longitude values if they are too far from valid range (:pull:`636`) by :user:`jvegreg`

Preprocessor
~~~~~~~~~~~~

-  Implemented concatenation of cubes with derived coordinates (:pull:`546`) by :user:`schlunma`
-  Fix derived variable ctotal calculation depending on project and standard name (:pull:`620`) by :user:`valeriupredoi`
-  State of the art FX variables handling without preprocessing (:pull:`557`) by :user:`valeriupredoi`
-  Add max, min and std operators to multimodel (:pull:`602`) by :user:`jvegreg`
-  Added preprocessor to extract amplitude of cycles (:pull:`597`) by :user:`schlunma`
-  Overhaul concatenation and allow for correct concatenation of multiple overlapping datasets (:pull:`615`) by :user:`valeriupredoi`
-  Change volume stats to handle and output masked array result (:pull:`618`) by :user:`valeriupredoi`
-  Area_weights for cordex in area_statistics (:pull:`631`) by :user:`mwjury`
-  Accept cubes as input in multimodel (:pull:`637`) by :user:`sloosvel`
-  Make multimodel work correctly with yearly data (:pull:`677`) by :user:`valeriupredoi`
-  Optimize time weights in time preprocessor for climate statistics (:pull:`684`) by :user:`valeriupredoi`
-  Add percentiles to multi-model stats (:pull:`679`) by :user:`Peter9192`

.. _changelog-v2-0-0b9:

v2.0.0b9
--------

This release includes

Bug fixes
~~~~~~~~~

-  Cast dtype float32 to output from zonal and meridional area preprocessors (:pull:`581`) by :user:`valeriupredoi`

Improvements
~~~~~~~~~~~~

-  Unpin on Python<3.8 for conda package (run) (:pull:`570`) by :user:`valeriupredoi`
-  Update pytest installation marker (:pull:`572`) by :user:`bouweandela`
-  Remove vmrh2o (:pull:`573`) by :user:`mattiarighi`
-  Restructure documentation (:pull:`575`) by :user:`bouweandela`
-  Fix mask in land variables for CCSM4 (:pull:`579`) by :user:`zklaus`
-  Fix derive scripts wrt required method (:pull:`585`) by :user:`zklaus`
-  Check coordinates do not have repeated standard names (:pull:`558`) by :user:`jvegreg`
-  Added derivation script for co2s (:pull:`587`) by :user:`schlunma`
-  Adapted custom co2s table to match CMIP6 version (:pull:`588`) by :user:`schlunma`
-  Increase version to v2.0.0b9 (:pull:`593`) by :user:`bouweandela`
-  Add a method to save citation information (:pull:`402`) by :user:`SarahAlidoost`

For older releases, see the release notes on https://github.com/ESMValGroup/ESMValCore/releases.
