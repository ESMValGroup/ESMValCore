These are custom coordinate variables that were used with the custom CMIP5
CMOR tables. They are problematic because they overwrite the standard CMIP5
coordinates and thereby modify the CMIP5 CMOR table when used together with
that table. They are kept here until config-developer.yml and the associated
`esmvalcore.cmor.table.CustomInfo` class is retired in v2.16.0.
