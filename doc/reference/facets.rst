.. _facets:

Facets
======

ESMValCore uses "facets" to search for and define input data, both in the
:ref:`recipe <recipe>` and in the :class:`esmvalcore.dataset.Dataset` object.
This allows specifying data without relying on e.g. file names or directory structures.

A facet is a key-value pair that describes a certain property of the data and
enables `faceted search <https://en.wikipedia.org/wiki/Faceted_search>`_. For
example, the ``dataset`` facet can be used to specify the climate model from
which data should be taken, e.g. ``dataset: HadGEM3-GC31-LM``.

ESMValCore uses its own set of facets which is consistent across all projects
it supports. A project is a collection of datasets that share certain
properties, e.g. CMIP7 or CORDEX. Each project has its own set of facets that
are relevant for that project.

Here is a mapping from the facet names used in ESMValCore to the corresponding
project specific facet names.

CMIP7
-----

`Official CMIP7 facets <https://wcrp-cmip.github.io/cmip7-guidance/CMIP7/global_attributes/>`__.

+------------------+-----------------------+
| ESMValCore facet | CMIP7 facet name      |
+==================+=======================+
| activity         | activity_id           |
+------------------+-----------------------+
| branding_suffix  | branding_suffix       |
+------------------+-----------------------+
| dataset          | source_id             |
+------------------+-----------------------+
| ensemble         | variant_label         |
+------------------+-----------------------+
| exp              | experiment_id         |
+------------------+-----------------------+
| frequency        | frequency             |
+------------------+-----------------------+
| grid             | grid_label            |
+------------------+-----------------------+
| institute        | institution_id        |
+------------------+-----------------------+
| realm            | realm                 |
+------------------+-----------------------+
| region           | region                |
+------------------+-----------------------+
| project          | mip_era               |
+------------------+-----------------------+
| short_name       | variable_id           |
+------------------+-----------------------+

CMIP6
-----

`Official CMIP6 facets <https://wcrp-cmip.github.io/WGCM_Infrastructure_Panel/Papers/CMIP6_global_attributes_filenames_CVs_v6.2.7.pdf>`__.

+------------------+-----------------------+
| ESMValCore facet | CMIP6 facet name      |
+==================+=======================+
| activity         | activity_id           |
+------------------+-----------------------+
| dataset          | source_id             |
+------------------+-----------------------+
| ensemble         | member_id             |
+------------------+-----------------------+
| exp              | experiment_id         |
+------------------+-----------------------+
| frequency        | frequency             |
+------------------+-----------------------+
| grid             | grid_label            |
+------------------+-----------------------+
| institute        | institution_id        |
+------------------+-----------------------+
| mip              | table_id              |
+------------------+-----------------------+
| realm            | realm                 |
+------------------+-----------------------+
| project          | mip_era               |
+------------------+-----------------------+
| short_name       | variable_id           |
+------------------+-----------------------+

CMIP5
-----

`Official CMIP5 facets <https://pcmdi.github.io/mips/cmip5/docs/CMIP5_output_metadata_requirements.pdf>`__.
Note that there appear to be difference between the official facets and those
used on ESGF. Below we present the values used on ESGF.

+------------------+-----------------------+
| ESMValCore facet | CMIP5 facet name      |
+==================+=======================+
| dataset          | model                 |
+------------------+-----------------------+
| ensemble         | ensemble              |
+------------------+-----------------------+
| exp              | experiment            |
+------------------+-----------------------+
| frequency        | time_frequency        |
+------------------+-----------------------+
| institute        | institute             |
+------------------+-----------------------+
| mip              | cmor_table            |
+------------------+-----------------------+
| realm            | realm                 |
+------------------+-----------------------+
| product          | product               |
+------------------+-----------------------+
| project          | project               |
+------------------+-----------------------+
| short_name       | variable              |
+------------------+-----------------------+


CMIP3
-----

+------------------+-----------------------+
| ESMValCore facet | CMIP3 facet name      |
+==================+=======================+
| dataset          | model                 |
+------------------+-----------------------+
| ensemble         | ensemble              |
+------------------+-----------------------+
| exp              | experiment            |
+------------------+-----------------------+
| frequency        | time_frequency        |
+------------------+-----------------------+
| short_name       | variable              |
+------------------+-----------------------+

CORDEX
-------

`Official CORDEX-CMIP5 facets <https://zenodo.org/records/15223120>`__.
Note that there appear to be difference between the official facets and those
used on ESGF. Below we present the values used on ESGF.

+------------------+-----------------------+
| ESMValCore facet | CORDEX facet name     |
+==================+=======================+
| dataset          | rcm_name              |
+------------------+-----------------------+
| driver           | driving_model         |
+------------------+-----------------------+
| domain           | domain                |
+------------------+-----------------------+
| ensemble         | ensemble              |
+------------------+-----------------------+
| exp              | experiment            |
+------------------+-----------------------+
| frequency        | time_frequency        |
+------------------+-----------------------+
| institute        | institute             |
+------------------+-----------------------+
| product          | product               |
+------------------+-----------------------+
| short_name       | variable              |
+------------------+-----------------------+

obs4MIPs
--------

`Official obs4MIPs facets <https://doi.org/10.5281/zenodo.11500473>`__.
Note that obs4MIPs first followed the CMIP5 conventions before switching to
the CMIP6 conventions. That means that both conventions are in use depending on
when a particular dataset was published. See the CMIP5 and CMIP6 tables above
for the mappings.
