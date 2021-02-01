"""Recipe metadata."""

import logging
import pprint
import textwrap
from pathlib import Path

import yaml

from esmvalcore._recipe import Recipe as RecipeEngine

from . import CFG
from ._logging import log_to_dir
from .recipe_metadata import Contributor, Project, Reference
from .recipe_output import RecipeOutput

logger = logging.getLogger(__file__)


class Recipe():
    """API wrapper for the esmvalcore Recipe object.

    This class can be used to inspect and run the recipe.

    Parameters
    ----------
    path : pathlike
        Path to the recipe.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f'Cannot find recipe: `{path}`.')

        self._data = None
        self._authors = None
        self._maintainers = None
        self._projects = None
        self._references = None
        self._description = None
        self._engine = None

    def __repr__(self) -> str:
        """Return canonical string representation."""
        return f'{self.__class__.__name__}({self.name!r})'

    def _repr_markdown_(self) -> str:
        """Represent using markdown renderer in a notebook environment."""
        return self.render('markdown')

    def __str__(self) -> str:
        """Return string representation."""
        return self.render('plaintext')

    def to_markdown(self) -> str:
        """Return markdown formatted string."""
        return self.render('markdown')

    def render(self, renderer: str = 'plaintext') -> str:
        """Return formatted string.

        Parameters
        ----------
        renderer : str
            Choose the renderer for the string representation.
            Must be one of 'plaintext', 'markdown'

        Returns
        -------
        str
            Rendered representation of the recipe documentation.
        """
        bullet = '\n - '
        string = f'## {self.name}'

        string += '\n\n'
        string += f'{self.description}'

        string += '\n\n### Authors'
        for author in self.authors:
            string += f'{bullet}{author}'

        string += '\n\n### Maintainers'
        for maintainer in self.maintainers:
            string += f'{bullet}{maintainer}'

        if self.projects:
            string += '\n\n### Projects'
            for project in self.projects:
                string += f'{bullet}{project}'

        if self.references:
            string += '\n\n### References'
            for reference in self.references:
                string += bullet + reference.render(renderer)

        string += '\n'

        return string

    @property
    def data(self) -> dict:
        """Return dictionary representation of the recipe."""
        if self._data is None:
            self._data = yaml.safe_load(open(self.path, 'r'))
        return self._data

    @property
    def name(self) -> str:
        """Name of the recipe."""
        return self.path.stem.replace('_', ' ').capitalize()

    @property
    def description(self) -> str:
        """Recipe description."""
        if self._description is None:
            description = self.data['documentation']['description']
            self._description = '\n'.join(textwrap.wrap(description))
        return self._description

    @property
    def authors(self) -> tuple:
        """List of recipe authors."""
        if self._authors is None:
            tags = self.data['documentation']['authors']
            self._authors = tuple(Contributor.from_tag(tag) for tag in tags)
        return self._authors

    @property
    def maintainers(self) -> tuple:
        """List of recipe maintainers."""
        if self._maintainers is None:
            tags = self.data['documentation'].get('maintainer', [])
            self._maintainers = tuple(
                Contributor.from_tag(tag) for tag in tags)
        return self._maintainers

    @property
    def projects(self) -> tuple:
        """List of recipe projects."""
        if self._projects is None:
            tags = self.data['documentation'].get('projects', [])
            self._projects = tuple(Project.from_tag(tag) for tag in tags)
        return self._projects

    @property
    def references(self) -> tuple:
        """List of project references."""
        if self._references is None:
            tags = self.data['documentation'].get('references', [])
            self._references = tuple(Reference.from_tag(tag) for tag in tags)
        return self._references

    def _load(self, session: dict):
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
        config_user = session.to_config_user()

        logger.info(pprint.pformat(config_user))

        self._engine = RecipeEngine(raw_recipe=self.data,
                                    config_user=config_user,
                                    recipe_file=self.path)

    def run(self, task: str = None, session: dict = None):
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
        if not session:
            session = CFG.start_session(self.path.stem)

        if task:
            session['diagnostics'] = task

        with log_to_dir(session.run_dir):
            self._load(session=session)
            self._engine.run()

        return self.get_output()

    def get_output(self) -> dict:
        """Get output from recipe.

        Returns
        -------
        output : dict
            Returns output of the recipe as instances of :obj:`OutputFile`
            grouped by diagnostic task.
        """
        if not self._engine:
            raise AttributeError('Run the recipe first using `.run()`.')

        raw_output = self._engine.get_product_output()

        return RecipeOutput(raw_output)
