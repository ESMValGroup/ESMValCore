Changelog
=========

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
