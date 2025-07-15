.. _derivation:

*******************
Deriving a variable
*******************

The variable derivation preprocessor module allows to derive variables which are
not in the CMIP standard data request using standard variables as input.
This is a special type of :ref:`preprocessor function <preprocessor_function>`.
All derivation scripts are located in
`esmvalcore/preprocessor/_derive/ <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/preprocessor/_derive>`_.
A typical example looks like this:

.. code-block:: py

   """Derivation of variable `dummy`."""

   from iris.cube import Cube, CubeList

   from esmvalcore.typing import Facets, FacetValue

   from ._baseclass import DerivedVariableBase


   class DerivedVariable(DerivedVariableBase):
       """Derivation of variable `dummy`."""

       @staticmethod
       def required(project: str) -> list[Facets]:
           """Declare the variables needed for derivation."""
           mip = "fx"
           if project == "CMIP6":
               mip = "Ofx"
           required = [
               {"short_name": "var_a"},
               {"short_name": "var_b", "mip": mip, "optional": True},
           ]
           return required

       @staticmethod
       def calculate(cubes: CubeList) -> Cube:
           """Compute `dummy`."""

           # `cubes` is a CubeList containing all required variables.
           cube = do_something_with(cubes)

           # Return single cube at the end
           return cube

The static function ``required(project)`` returns a :obj:`list` of :obj:`~esmvalcore.typing.Facets`
containing all required variables for deriving the derived variable. Its only
argument is the ``project`` of the specific dataset. In this particular
example script, the derived variable ``dummy`` is derived from ``var_a`` and
``var_b``. It is possible to specify arbitrary attributes for each required
variable, e.g. ``var_b`` uses the mip ``fx`` (or ``Ofx`` in the case of
CMIP6) instead of the original one of ``dummy``. Note that you can also declare
a required variable as ``optional=True``, which allows the skipping of this
particular variable during data extraction. For example, this is useful for
fx variables which are often not available for observational datasets.
Otherwise, the tool will fail if not all required variables are available for
all datasets.

The actual derivation takes place in the static function ``calculate(cubes)``
which returns a single :class:`~iris.cube.Cube` containing the derived
variable. Its only argument ``cubes`` is a :class:`~iris.cube.CubeList`
containing all required variables.

If no MIP table entry for the derived variable exists for the given ``mip``,
the tool will also look in other ``mip`` tables for the same ``project`` to find
the definition of derived variables. To contribute a completely new derived
variable, it is necessary to define a name for it and to provide the
corresponding CMOR table. This is to guarantee the proper metadata definition
is attached to the derived data. Such custom CMOR tables are collected as part
of the `ESMValCore package <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables/custom>`_.
