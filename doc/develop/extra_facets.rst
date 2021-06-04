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
