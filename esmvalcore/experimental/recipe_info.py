import textwrap
from pathlib import Path

from .recipe_metadata import Contributor, Project, Reference
from .templates import get_template


class RecipeInfo():
    """API wrapper for the esmvalcore Recipe object.

    This class can be used to inspect and run the recipe.

    Parameters
    ----------
    filename : pathlike
        Path to the recipe.
    """
    def __init__(self, data, filename: str = None):
        self.filename = Path(filename)
        self.data = data
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
        return self.info.render('markdown')

    def _repr_html_(self) -> str:
        """Represent using html renderer in a notebook environment."""
        return self.info.render('markdown')

    def __str__(self) -> str:
        """Return string representation."""
        return self.info.render('plaintext')

    @property
    def name(self) -> str:
        """Name of the recipe."""
        return self.filename.stem.replace('_', ' ').capitalize()

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

    def to_html(self):
        """Render info object to html."""
        template = get_template('recipe_info.j2')
        return template.render(info=self)

    def render(self, renderer: str = 'plaintext') -> str:
        """Return formatted string.

        Parameters
        ----------
        renderer : str
            Choose the renderer for the string representation.
            Must be one of 'plaintext', 'markdown', 'html'

        Returns
        -------
        str
            Rendered representation of the recipe documentation.
        """
        if renderer == 'html':
            return self.to_html()

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
