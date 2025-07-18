"""Recipe metadata."""

import logging
import os
import pprint
import shutil
from pathlib import Path

import yaml

from esmvalcore._recipe.recipe import Recipe as RecipeEngine
from esmvalcore.config import CFG, Session

from ._logging import log_to_dir
from .recipe_info import RecipeInfo
from .recipe_output import RecipeOutput

logger = logging.getLogger(__name__)


class Recipe:
    """API wrapper for the esmvalcore Recipe object.

    This class can be used to inspect and run the recipe.

    Parameters
    ----------
    path : pathlike
        Path to the recipe.
    """

    def __init__(self, path: os.PathLike):
        self.path = Path(path)
        if not self.path.exists():
            msg = f"Cannot find recipe: `{path}`."
            raise FileNotFoundError(msg)

        self._engine: RecipeEngine | None = None
        self._data: dict | None = None
        self.last_session: Session | None = None
        self.info = RecipeInfo(self.data, filename=self.path.name)

    def __repr__(self) -> str:
        """Return canonical string representation."""
        return f"{self.__class__.__name__}({self.name!r})"

    def __str__(self) -> str:
        """Return string representation."""
        return str(self.info)

    def _repr_html_(self) -> str:
        """Return html representation."""
        return self.render()

    def render(self, template=None):
        """Render output as html.

        template : :obj:`Template`
            An instance of :py:class:`jinja2.Template` can be passed to
            customize the output.
        """
        return self.info.render(template=template)

    @property
    def name(self):
        """Return the name of the recipe."""
        return self.info.name

    @property
    def data(self) -> dict:
        """Return dictionary representation of the recipe."""
        if self._data is None:
            with open(self.path, encoding="utf-8") as yaml_file:
                self._data = yaml.safe_load(yaml_file)
        return self._data

    def _load(self, session: Session) -> RecipeEngine:
        """Load the recipe.

        This method loads the recipe into the internal ESMValCore Recipe
        format.

        Parameters
        ----------
        session : :obj:`Session`
            Defines the config parameters and location where the recipe
            output will be stored. If ``None``, a new session will be
            started automatically.

        Returns
        -------
        recipe : :obj:`esmvalcore._recipe.Recipe`
            Return an instance of the Recipe
        """
        logger.info(pprint.pformat(session))

        return RecipeEngine(
            raw_recipe=self.data,
            session=session,
            recipe_file=self.path,
        )

    def run(
        self,
        task: str | None = None,
        session: Session | None = None,
    ):
        """Run the recipe.

        This function loads the recipe into the ESMValCore recipe format
        and runs it.

        Parameters
        ----------
        task : str
            Specify the name of the diagnostic or preprocessor to run a
            single task.
        session : :obj:`Session`, optional
            Defines the config parameters and location where the recipe
            output will be stored. If ``None``, a new session will be
            started automatically.

        Returns
        -------
        output : dict
            Returns output of the recipe as instances of :obj:`OutputItem`
            grouped by diagnostic task.
        """
        if session is None:
            session = CFG.start_session(self.path.stem)

        self.last_session = session

        if task:
            session["diagnostics"] = task

        with log_to_dir(session.run_dir):
            self._engine = self._load(session=session)
            self._engine.run()

        shutil.copy2(self.path, session.run_dir)

        output = self.get_output()
        output.write_html()

        return output

    def get_output(self) -> RecipeOutput:
        """Get output from recipe.

        Returns
        -------
        output : dict
            Returns output of the recipe as instances of :obj:`OutputFile`
            grouped by diagnostic task.
        """
        if self._engine is None:
            msg = "Run the recipe first using `.run()`."
            raise AttributeError(msg)

        output = self._engine.get_output()
        task_output = output["task_output"]

        return RecipeOutput(
            task_output=task_output,
            session=self.last_session,
            info=self.info,
        )
