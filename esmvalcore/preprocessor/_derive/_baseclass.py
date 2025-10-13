"""Contains the base class for derived variables."""

from abc import abstractmethod

from iris.cube import Cube, CubeList

from esmvalcore.typing import Facets


class DerivedVariableBase:
    """Base class for derived variables."""

    @staticmethod
    @abstractmethod
    def required(project: str) -> list[Facets]:
        """Return required variables for derivation.

        This method needs to be overridden in the child class belonging to the
        desired variable to derive.

        Note
        ----
        It is possible to declare a required variable as ``optional=True``,
        which allows the skipping of this particular variable during data
        extraction. For example, this is useful for fx variables which are
        often not available for observational datasets. Otherwise, the tool
        will fail if not all required variables are available for all datasets.

        Parameters
        ----------
        project:
            Project of the dataset for which the desired variable is derived.

        Returns
        -------
        list[esmvalcore.typing.Facets]
            List of facets.

        """

    @staticmethod
    @abstractmethod
    def calculate(cubes: CubeList) -> Cube:
        """Compute desired derived variable.

        This method needs to be overridden in the child class belonging to the
        desired variable to derive.

        Parameters
        ----------
        cubes:
            Includes all the needed variables (incl. fx variables) for
            derivation defined in ``required``.

        Returns
        -------
        iris.cube.Cube
            New derived variable.

        """
