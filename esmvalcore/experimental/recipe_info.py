"""Recipe metadata."""

import textwrap
from pathlib import Path

import pybtex
import yaml
from pybtex.database.input import bibtex

from esmvalcore._citation import REFERENCES_PATH
from esmvalcore._config import TAGS


class RenderError(BaseException):
    """Error during rendering of object."""


class Contributor:
    """Contains contributor (author or maintainer) information."""

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

    def _repr_markdown_(self):
        """Represent using markdown renderer in a notebook environment."""
        return str(self)

    @classmethod
    def from_tag(cls, tag: str):
        """Return an instance of Contributor from a tag (``TAGS``).

        Contributors are defined by author tags in ``config-
        references.yml``.
        """
        mapping = TAGS['authors'][tag]

        name = ' '.join(reversed(mapping['name'].split(', ')))
        institute = mapping.get('institute', 'No affiliation')
        orcid = mapping['orcid']

        return cls(name=name, institute=institute, orcid=orcid)


class Project:
    """Contains author information."""

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
    def from_tag(cls, tag: str):
        """Return an instance of Project from a tag (``TAGS``).

        The project tags are defined in ``config-references.yml``.
        """
        project = TAGS['projects'][tag]
        return cls(project=project)


class Reference:
    """Contains reference information."""

    def __init__(self, filename):
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
    def from_tag(cls, tag: str):
        """Return an instance of Reference from a bibtex tag.

        The bibtex tags resolved as
        ``esmvaltool/references/{tag}.bibtex``.
        """
        filename = Path(REFERENCES_PATH, f'{tag}.bibtex')
        return cls(filename)

    def __repr__(self):
        """Return canonical string representation."""
        return f'{self.__class__.__name__}({self._key!r})'

    def __str__(self):
        """Return string representation."""
        return self.render(renderer='plaintext')

    def _repr_markdown_(self):
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
        """
        bullet = '\n - '
        string = f'## {self.name}'

        string += '\n\n'
        string += f'{self.description}'

        string += '\n\n### Contributors'
        for author in self.authors:
            string += f'{bullet}{author}'

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
            tags = self.data['documentation']['maintainer']
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
