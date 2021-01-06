"""Recipe metadata."""

import logging
import pprint
import textwrap
from pathlib import Path

import pybtex
import yaml
from pybtex.database.input import bibtex

from esmvalcore._citation import REFERENCES_PATH
from esmvalcore._config import TAGS

from ._logging import log_to_dir

logger = logging.getLogger(__file__)


class RenderError(BaseException):
    """Error during rendering of object."""


class Contributor:
    """Contains contributor (author or maintainer) information.".

    Parameters
    ----------
    name : str
        Name of the author, i.e. ``'John Doe'``
    institute : str
        Name of the institute
    orcid : str, optional
        ORCID url
    """
    def __init__(self, name: str, institute: str, orcid: str = None):
        self.name = name
        self.institute = institute
        self.orcid = orcid

    def __repr__(self) -> str:
        """Return canonical string representation."""
        return (f'{self.__class__.__name__}({self.name!r},'
                f' institute={self.institute!r}, orcid={self.orcid!r})')

    def __str__(self) -> str:
        """Return string representation."""
        string = f'{self.name} ({self.institute}'
        if self.orcid:
            string += f'; {self.orcid}'
        string += ')'
        return string

    def _repr_markdown_(self) -> str:
        """Represent using markdown renderer in a notebook environment."""
        return str(self)

    @classmethod
    def from_tag(cls, tag: str) -> 'Contributor':
        """Return an instance of Contributor from a tag (``TAGS``).

        Parameters
        ----------
        tag : str
            The contributor tags are defined in the authors section in
            ``config-references.yml``.
        """
        mapping = TAGS['authors'][tag]

        name = ' '.join(reversed(mapping['name'].split(', ')))
        institute = mapping.get('institute', 'No affiliation')
        orcid = mapping['orcid']

        return cls(name=name, institute=institute, orcid=orcid)


class Project:
    """Contains project information.

    Parameters
    ----------
    project : str
        Defines the project title.
    """
    def __init__(self, project: str):
        self.project = project

    def __repr__(self) -> str:
        """Return canonical string representation."""
        return f'{self.__class__.__name__}({self.project!r})'

    def __str__(self) -> str:
        """Return string representation."""
        string = f'{self.project}'
        return string

    @classmethod
    def from_tag(cls, tag: str) -> 'Project':
        """Return an instance of Project from a tag (``TAGS``).

        Parameters
        ----------
        tag : str
            The project tags are defined in ``config-references.yml``.
        """
        project = TAGS['projects'][tag]
        return cls(project=project)


class Reference:
    """Contains reference information.

    Parameters
    ----------
    filename : str
        Name of the bibtex file.

    Raises
    ------
    NotImplementedError
        If the bibtex file contains more than 1 entry.
    """
    def __init__(self, filename: str):
        parser = bibtex.Parser(strict=False)
        bib_data = parser.parse_file(filename)

        if len(bib_data.entries) > 1:
            raise NotImplementedError(
                f'{self.__class__.__name__} cannot handle bibtex files '
                'with more than 1 entry.')

        self._bib_data = bib_data
        self._key, self._entry = list(bib_data.entries.items())[0]
        self._filename = filename

    @classmethod
    def from_tag(cls, tag: str) -> 'Reference':
        """Return an instance of Reference from a bibtex tag.

        Parameters
        ----------
        tag : str
            The bibtex tags resolved as ``esmvaltool/references/{tag}.bibtex``
            or the corresponding directory as defined by the diagnostics path.
        """
        filename = Path(REFERENCES_PATH, f'{tag}.bibtex')
        return cls(filename)

    def __repr__(self) -> str:
        """Return canonical string representation."""
        return f'{self.__class__.__name__}({self._key!r})'

    def __str__(self) -> str:
        """Return string representation."""
        return self.render(renderer='plaintext')

    def _repr_markdown_(self) -> str:
        """Represent using markdown renderer in a notebook environment."""
        return self.render(renderer='markdown')

    def render(self, renderer: str = 'plaintext') -> str:
        """Render the reference.

        Parameters
        ----------
        renderer : str
            Choose the renderer for the string representation.
            Must be one of: 'plaintext', 'markdown', 'html', 'latex'

        Returns
        -------
        str
            Rendered reference
        """
        style = 'plain'  # alpha, plain, unsrt, unsrtalpha
        backend = pybtex.plugin.find_plugin('pybtex.backends', renderer)()
        style = pybtex.plugin.find_plugin('pybtex.style.formatting', style)()

        try:
            formatter = style.format_entry(self._key, self._entry)
            rendered = formatter.text.render(backend)
        except Exception as err:
            raise RenderError(
                f'Could not render {self._key!r}: {err}') from None

        return rendered


class RecipeInfo():
    """Contains Recipe metadata.

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

    def load(self, session: dict):
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

        from esmvalcore._recipe import Recipe
        recipe = Recipe(raw_recipe=self.data,
                        config_user=config_user,
                        recipe_file=self.path)

        return recipe

    def run(self, session: dict = None):
        """Run the recipe.

        This function loads the recipe into the ESMValCore recipe format
        and runs it.

        Parameters
        ----------
        session : :obj:`Session`, optional
            Defines the config parameters and location where the recipe
            output will be stored. If ``None``, a new session will be
            started automatically.

        Returns
        -------
        output : None
            Output of the recipe (Not implemented yet)
        """
        if not session:
            from . import CFG
            session = CFG.start_session(self.path.stem)

        with log_to_dir(session.run_dir):
            recipe = self.load(session=session)
            recipe.run()

        out = {}
        for task in recipe.tasks:
            out[task.name] = task.settings

        return out
