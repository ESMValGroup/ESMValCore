"""Handles recipe metadata (under 'documentation' section)."""
import os
import textwrap
from pathlib import Path
from typing import Optional, Tuple, Union

import yaml

from .recipe_metadata import Contributor, Project, Reference
from .templates import get_template


class RecipeInfo():
    """API wrapper for the esmvalcore Recipe object.

    This class can be used to inspect and run the recipe.

    Parameters
    ----------
    filename : pathlike
        Name of recipe file
    """

    def __init__(self, data, filename: Union[os.PathLike, str]):
        self.filename = Path(filename).name
        self.data = data
        self._authors: Optional[Tuple[Contributor, ...]] = None
        self._maintainers: Optional[Tuple[Contributor, ...]] = None
        self._projects: Optional[Tuple[Project, ...]] = None
        self._references: Optional[Tuple[Reference, ...]] = None
        self._title: Optional[str] = None
        self._description: Optional[str] = None

    def __repr__(self) -> str:
        """Return canonical string representation."""
        return f'{self.__class__.__name__}({self.name!r})'

    def __str__(self) -> str:
        """Return string representation."""
        bullet = '\n - '
        string = f'## {self.title}'

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
                string += bullet + reference.render('plaintext')

        string += '\n'

        return string

    def _repr_html_(self) -> str:
        """Represent using html renderer in a notebook environment."""
        return self.render()

    @classmethod
    def from_yaml(cls, path: str):
        """Return instance of 'RecipeInfo' from a recipe in yaml format."""
        data = yaml.safe_load(open(path, 'r'))
        return cls(data, filename=path)

    @property
    def name(self) -> str:
        """Name of the recipe."""
        return Path(self.filename).stem.replace('_', ' ').capitalize()

    @property
    def title(self) -> str:
        """Title of the recipe."""
        if self._title is None:
            self._title = self.data['documentation']['title']
        return self._title

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
            tags = self.data['documentation'].get('authors', ())
            self._authors = tuple(Contributor.from_tag(tag) for tag in tags)
        return self._authors

    @property
    def maintainers(self) -> tuple:
        """List of recipe maintainers."""
        if self._maintainers is None:
            tags = self.data['documentation'].get('maintainer', ())
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

    def render(self, template=None):
        """Render output as html.

        template : :obj:`Template`
            An instance of :py:class:`jinja2.Template` can be passed to
            customize the output.
        """
        if not template:
            template = get_template(self.__class__.__name__ + '.j2')
        rendered = template.render(info=self)

        return rendered

    def resolve(self) -> None:
        """Force resolve of all tags in recipe.

        Raises
        ------
        LookupError
            Raised when some tags in the recipe cannot be resolved
        """
        try:
            # calling these attributes forces the tags to be resolved.
            _ = self.authors
            _ = self.maintainers
            _ = self.projects
            _ = self.references
        except BaseException as error:
            message = f"Some tags in the recipe could not be resolved: {error}"
            raise LookupError(message) from error
