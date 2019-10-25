"""Contains the base class for derived variables."""
from abc import abstractmethod


class DerivedVariableBase:
    """Base class for derived variables."""

    @staticmethod
    @abstractmethod
    def required(project):
        """Return required variables for derivation.

        This method needs to be overridden in the child class belonging to the
        desired variable to derive.

        Note
        ----
        It is possible to declare a required variable as `optional=True`, which
        allows the skipping of this particular variable during data extraction.
        For example, this is useful for fx variables which are often not
        available for observational datasets. Otherwise, the tool will fail if
        not all required variables are available for all datasets.

        Parameters
        ----------
        project : str
            Project of the dataset for which the desired variable is derived.

        Returns
        -------
        list of dict
            List of variable metadata.

        """

    @staticmethod
    @abstractmethod
    def calculate(cubes):
        """Compute desired derived variable.

        This method needs to be overridden in the child class belonging to the
        desired variable to derive.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Includes all the needed variables (incl. fx variables) for
            derivation defined in the static class variable
            `_required_variables`.

        Returns
        -------
        iris.cube.Cube
            New derived variable.

        Raises
        ------
        NotImplementedError
            If the desired variable derivation is not implemented, i.e. if this
            method is called from this base class and not a child class.

        """
