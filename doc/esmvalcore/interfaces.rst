.. _interfaces:

Diagnostic script interfaces
============================

In order to communicate with diagnostic scripts, ESMValCore uses YAML files.
The YAML files provided by ESMValCore to the diagnostic script tell the diagnostic script the settings that were provided in the recipe and where to find the pre-processed input data.
On the other hand, the YAML file provided by the diagnostic script to ESMValCore tells ESMValCore which pre-processed data was used to create what plots.
The latter is optional, but needed for recording provenance.

Provenance
----------
When ESMValCore (the ``esmvaltool`` command) runs a recipe, it will first find all data and run the default preprocessor steps plus any
additional preprocessing steps defined in the recipe. Next it will run the diagnostic script defined in the recipe
and finally it will store provenance information. Provenance information is stored in the
`W3C PROV XML format <https://www.w3.org/TR/prov-xml/>`_.
To read in and extract information, or to plot these files, the
`prov <https://prov.readthedocs.io>`_ Python package can be used.
In addition to provenance information, a caption is also added to the plots.

.. _interface_esmvalcore_diagnostic:

Information provided by ESMValCore to the diagnostic script
-----------------------------------------------------------
To provide the diagnostic script with the information it needs to run (e.g. location of input data, various settings),
the ESMValCore creates a YAML file called settings.yml and provides the path to this file as the first command line
argument to the diagnostic script.

The most interesting settings provided in this file are

.. code-block:: yaml

  run_dir:  /path/to/recipe_output/run/diagnostic_name/script_name
  work_dir: /path/to/recipe_output/work/diagnostic_name/script_name
  plot_dir: /path/to/recipe_output/plots/diagnostic_name/script_name
  input_files:
    - /path/to/recipe_output/preproc/diagnostic_name/ta/metadata.yml
    - /path/to/recipe_output/preproc/diagnostic_name/pr/metadata.yml

Custom settings in the script section of the recipe will also be made available in this file.

There are three directories defined:

- :code:`run_dir` use this for storing temporary files
- :code:`work_dir` use this for storing NetCDF files containing the data used to make a plot
- :code:`plot_dir` use this for storing plots

Finally :code:`input_files` is a list of YAML files, containing a description of the preprocessed data. Each entry in these
YAML files is a path to a preprocessed file in NetCDF format, with a list of various attributes.
An example preprocessor metadata.yml file could look like this:

.. code-block:: yaml

  ? /path/to/recipe_output/preproc/diagnostic_name/pr/CMIP5_GFDL-ESM2G_Amon_historical_r1i1p1_T2Ms_pr_2000-2002.nc
  : alias: GFDL-ESM2G
    cmor_table: CMIP5
    dataset: GFDL-ESM2G
    diagnostic: diagnostic_name
    end_year: 2002
    ensemble: r1i1p1
    exp: historical
    filename: /path/to/recipe_output/preproc/diagnostic_name/pr/CMIP5_GFDL-ESM2G_Amon_historical_r1i1p1_T2Ms_pr_2000-2002.nc
    frequency: mon
    institute: [NOAA-GFDL]
    long_name: Precipitation
    mip: Amon
    modeling_realm: [atmos]
    preprocessor: preprocessor_name
    project: CMIP5
    recipe_dataset_index: 1
    reference_dataset: MPI-ESM-LR
    short_name: pr
    standard_name: precipitation_flux
    start_year: 2000
    units: kg m-2 s-1
    variable_group: pr
  ? /path/to/recipe_output/preproc/diagnostic_name/pr/CMIP5_MPI-ESM-LR_Amon_historical_r1i1p1_T2Ms_pr_2000-2002.nc
  : alias: MPI-ESM-LR
    cmor_table: CMIP5
    dataset: MPI-ESM-LR
    diagnostic: diagnostic_name
    end_year: 2002
    ensemble: r1i1p1
    exp: historical
    filename: /path/to/recipe_output/preproc/diagnostic1/pr/CMIP5_MPI-ESM-LR_Amon_historical_r1i1p1_T2Ms_pr_2000-2002.nc
    frequency: mon
    institute: [MPI-M]
    long_name: Precipitation
    mip: Amon
    modeling_realm: [atmos]
    preprocessor: preprocessor_name
    project: CMIP5
    recipe_dataset_index: 2
    reference_dataset: MPI-ESM-LR
    short_name: pr
    standard_name: precipitation_flux
    start_year: 2000
    units: kg m-2 s-1
    variable_group: pr


.. _interface_diagnostic_esmvalcore:

Information provided by the diagnostic script to ESMValCore
-----------------------------------------------------------

After the diagnostic script has finished running, ESMValCore will try to store provenance information. In order to
link the produced files to input data, the diagnostic script needs to store a YAML file called :code:`diagnostic_provenance.yml`
in its :code:`run_dir`.

For every output file (netCDF files, plot files, etc.) produced by the diagnostic script, there should be an entry in the :code:`diagnostic_provenance.yml` file.
The name of each entry should be the path to the file.
Each output file entry should at least contain the following items:

- :code:`ancestors` a list of input files used to create the plot.
- :code:`caption` a caption text for the plot.

Each file entry can also contain items from the categories defined in the file :code:`esmvaltool/config_references.yml`.
The short entries will automatically be replaced by their longer equivalent in the final provenance records.
It is possible to add custom provenance information by adding custom items to entries.

An example :code:`diagnostic_provenance.yml` file could look like this

.. code-block:: yaml

  ? /path/to/recipe_output/work/diagnostic_name/script_name/CMIP5_GFDL-ESM2G_Amon_historical_r1i1p1_pr_2000-2002_mean.nc
  : ancestors:[/path/to/recipe_output/preproc/diagnostic_name/pr/CMIP5_GFDL-ESM2G_Amon_historical_r1i1p1_pr_2000-2002.nc]
    authors: [andela_bouwe, righi_mattia]
    caption: Average Precipitation between 2000 and 2002 according to GFDL-ESM2G.
    domains: [global]
    plot_types: [zonal]
    references: [acknow_project]
    statistics: [mean]

  ? /path/to/recipe_output/plots/diagnostic_name/script_name/CMIP5_GFDL-ESM2G_Amon_historical_r1i1p1_pr_2000-2002_mean.png
  : ancestors:[/path/to/recipe_output/preproc/diagnostic_name/pr/CMIP5_GFDL-ESM2G_Amon_historical_r1i1p1_pr_2000-2002.nc]
    authors: [andela_bouwe, righi_mattia]
    caption: Average Precipitation between 2000 and 2002 according to GFDL-ESM2G.
    domains: [global]
    plot_types: ['zonal']
    references: [acknow_project]
    statistics: [mean]

You can check whether your diagnostic script successfully provided the provenance information to the ESMValCore by
checking the following points:

  - for each output file in the ``work_dir`` and ``plot_dir``, a file with the same
    name, but ending with ``_provenance.xml`` is created
  - the output file is shown on the ``index.html`` page
  - there were no warning messages in the log related to provenance

See :ref:`esmvaltool:recording-provenance` for more extensive usage notes.
