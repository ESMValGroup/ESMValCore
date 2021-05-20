Changelog
=========

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
-  Make CircleCI badge specific to master branch (`#902 <https://github.com/ESMValGroup/ESMValCore/pull/902>`__) `Bouwe Andela <https://github.com/bouweandela>`__
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
-  Fix duplication in preprocessors DEFAULT_ORDER introduced in `#696 <https://github.com/ESMValGroup/ESMValCore/pull/696>`__  (`#973 <https://github.com/ESMValGroup/ESMValCore/pull/973>`__) `Javier Vegas-Regidor <https://github.com/jvegasbsc>`__
-  Use consistent precision in multi-model statistics calculation and update reference data for tests (`#941 <https://github.com/ESMValGroup/ESMValCore/pull/941>`__) `Peter Kalverla <https://github.com/Peter9192>`__
-  Refactor multi-model statistics code to facilitate ensemble stats and lazy evaluation (`#949 <https://github.com/ESMValGroup/ESMValCore/pull/949>`__) `Peter Kalverla <https://github.com/Peter9192>`__
-  Add option to exclude input cubes in output of multimodel statistics to solve an issue introduced by `#949 <https://github.com/ESMValGroup/ESMValCore/pull/949>`__ (`#978 <https://github.com/ESMValGroup/ESMValCore/pull/978>`__) `Peter Kalverla <https://github.com/Peter9192>`__


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
