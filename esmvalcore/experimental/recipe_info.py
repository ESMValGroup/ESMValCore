"""Recipe metadata."""

import textwrap
from pathlib import Path

import yaml

from esmvalcore._citation import REFERENCES_PATH
from esmvalcore._config import TAGS

# TODO: Look into `functools.cached_property` for lazy evaluation (python 3.8+)


class Author:
    """Contains author information."""
    def __init__(self, name: str, institute: str, orcid: str = None):
        self.name = name
        self.institute = institute
        self.orcid = orcid

    def __repr__(self):
        """Return canonical string representation."""
        return (f'{self.__class__.__name__}({self.name!r},'
                ' institute={self.institute!r}, orcid={self.orcid!r})')

    def __str__(self):
        """Return string representation."""
        s = f'{self.name} ({self.institute})'
        if self.orcid:
            s += f'\n{self.orcid}'
        return s

    @classmethod
    def from_tag(cls, tag: str):
        """Return an instance of Author from a tag (TAGS).

        The author tags are defined in `config-references.yml`.
        """
        mapping = TAGS['authors'][tag]

        name = ' '.join(reversed(mapping['name'].split(', ')))
        institute = mapping.get('institute', 'No affiliation')
        orcid = mapping['orcid']

        return cls(name=name, institute=institute, orcid=orcid)


class Project:
    """Contains author information."""
    def __init__(self, project):
        self.project = project

    def __repr__(self):
        """Return canonical string representation."""
        return f'{self.__class__.__name__}({self.project!r})'

    def __str__(self):
        """Return string representation."""
        s = f'{self.project}'
        return s

    @classmethod
    def from_tag(cls, tag):
        """Return an instance of Project from a tag (TAGS).

        The project tags are defined in `config-references.yml`.
        """
        project = TAGS['projects'][tag]
        return cls(project=project)


class RecipeInfo():
    """Contains Recipe metadata."""
    def __init__(self, path):
        self.path = path

        self._mapping = None
        self._authors = None
        self._projects = None
        self._references = None
        self._description = None

    def __repr__(self):
        """Return canonical string representation."""
        return f'{self.__class__.__name__}({str(self.path)!r})'

    def __str__(self):
        """Return string representation."""
        return self.to_markdown()

    def to_markdown(self) -> str:
        """Return markdown formatted string."""
        s = f'## {self.name}'

        s += '\n\n'
        s += f'{self.description}'

        s += '\n\n### Authors'
        for author in self.authors:
            s += f'\n- {author}'

        if self.projects:
            s += '\n\n### Projects'
            for project in self.projects:
                s += f'\n- {project}'

        if self.references:
            s += '\n\n### References'
            for reference in self.references:
                s += f'\n- {reference}'

        s += '\n'

        return s

    @property
    def mapping(self):
        if self._mapping is None:
            self._mapping = yaml.safe_load(open(self.path, 'r'))
        return self._mapping

    @property
    def name(self):
        """Name of the recipe."""
        return self.path.stem.replace('_', ' ').capitalize()

    @property
    def description(self) -> str:
        """Recipe description."""
        if self._description is None:
            description = self.mapping['documentation']['description']
            self._description = '\n'.join(textwrap.wrap(description))
        return self._description

    @property
    def authors(self) -> tuple:
        """List of recipe authors."""
        if self._authors is None:
            tags = self.mapping['documentation']['authors']
            self._authors = tuple(Author.from_tag(tag) for tag in tags)
        return self._authors

    @property
    def projects(self) -> tuple:
        """List of recipe projects."""
        if self._projects is None:
            tags = self.mapping['documentation'].get('projects', [])
            self._projects = tuple(Project.from_tag(tag) for tag in tags)
        return self._projects

    @property
    def references(self) -> tuple:
        """List of project references."""
        if self._references is None:
            references = self.mapping['documentation'].get('references', [])
            self._references = tuple(
                Path(REFERENCES_PATH, f'{reference}.bibtex')
                for reference in references)
        return self._references
