.. _extra_facets:

************
Extra Facets
************

Sometimes it is useful to provide extra information for the loading of data,
particularly in the case of native model data, or observational or other data,
that generally follows the established standards, but is not part of the big
supported projects like CMIP, CORDEX, obs4MIPs.

To support this, we provide the extra facets facilities. Facets are the
key-value pairs described in :ref:`Datasets`. Extra facets allows for the
addition of more details per project, dataset, mip table, and variable name.

More precisely, one can provide this information in an extra yaml file, named
`{project}-something.yml`, where `{project}` corresponds to the project as used
by ESMValTool in :ref:`Datasets` and "something" is arbitrary.

Format of the extra facets files
================================
The extra facets are given in a yaml file, whose file name identifies the
project. Inside the file there is a hierarchy of nested dictionaries with the
following levels. At the top there is the `dataset` facet, followed by the `mip`
table, and finally the `short_name`. The leaf dictionary placed here gives the
extra facets that will be made available to data finder and the fix
infrastructure. The following example illustrates the concept.

.. _extra-facets-example-1:

.. code-block:: yaml
   :caption: Extra facet example file `native6-era5.yml`

   era5:
     Amon:
       tas: {file_var_name: "t2m", name_in_filename: "2m_temperature"}


Location of the extra facets files
==================================
Extra facets files can be placed in several different places. When we use them
to support a particular use-case within the ESMValTool project, they will be
provided in the sub-folder `extra_facets` inside the package
`esmvalcore._config`. If they are used from the user side, they can be either
placed in `~/.esmvaltool/extra_facets` or in any other directory of the users
choosing. In that case this directory must be added to the `config-user.yml`
file under the `extra_facets_dir` setting, which can take a single directory or
a list of directories.

The order in which the directories are searched is

1. The internal directory `esmvalcore._config/extra_facets`
2. The default user directory `~/.esmvaltool/extra_facets`
3. The custom user directories in the order in which they are given in
   `config-user.yml`.

The extra facets files within each of these directories are processed in
lexicographical order according to their file name.

In all cases it is allowed to supersede information from earlier files in later
files. This makes it possible for the user to effectively override even internal
default facets, for example to deal with local particularities in the data
handling.

Use of extra facets
===================
For extra facets to be useful, the information that they provide must be
applied. There are fundamentally two places where this comes into play. One is
the datafinder, the other are fixes.

Use of extra facets in the datafinder
-------------------------------------
Extra facets can be used to locate data files within the datafinder
framework. This is useful to build paths for directory structures and file names
that follow a different system than the established DRS for, e.g. CMIP.
A common application is the location of variables in multi-variable files as
often found in climate models' native output formats.

Another use case is files that use different names for variables in their
file name than for the netCDF4 variable name.

To apply the extra facets for this purpose, simply use the corresponding tag in
the applicable DRS inside the `config-developer.yml` file. For example, given
the extra facets in :ref:`extra-facets-example-1`, one might write the
following.

.. extra-facets-example-2:

.. code-block:: yaml
   :caption: Example drs use in `config-developer.yml`

   native6:
     input_file:
       default: '{name_in_filename}*.nc'

The same replacement mechanism can be employed everywhere where tags can be
used, particularly in `input_dir` and `input_file`.

Use of extra facets in fixes
----------------------------
In fixes, extra facets can be used to mold data into the form required by the
applicable standard. For example, if the input data is part of an observational
product that delivers surface temperature with a variable name of `t2m` inside a
file named `2m_temperature_1950_monthly.nc`, but the same variable is called
`tas` in the applicable standard, a fix can be created that reads the original
variable from the correct file, and provides a renamed variable to the rest of
the processing chain.

Normally, the applicable standard for variables is CMIP6.

For more details, refer to existing uses of this feature as examples.
