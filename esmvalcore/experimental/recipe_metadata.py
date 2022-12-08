"""API for recipe metadata."""

import pybtex
from pybtex.database.input import bibtex

from esmvalcore._config import DIAGNOSTICS, TAGS


class RenderError(BaseException):
    """Error during rendering of object."""


class Contributor:
    """Contains contributor (author or maintainer) information.

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
        mapping = TAGS.get_tag_value(section='authors', tag=tag)

        name = ' '.join(reversed(mapping['name'].split(', ')))
        institute = mapping.get('institute', 'No affiliation')
        orcid = mapping['orcid']

        return cls(name=name, institute=institute, orcid=orcid)

    @classmethod
    def from_dict(cls, attributes):
        """Return an instance of Contributor from a dictionary.

        Parameters
        ----------
        attributes : dict
            Dictionary containing name / institute [/ orcid].
        """
        name = attributes['name']
        institute = attributes['institute']
        orcid = attributes.get('orcid', None)
        return cls(name=name, institute=institute, orcid=orcid)


class Project:
    """Use this class to acknowledge a project associated with the recipe.

    Parameters
    ----------
    project : str
        The project title.
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
    """Parse reference information from bibtex entries.

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
        filename = DIAGNOSTICS.references / f'{tag}.bibtex'
        return cls(filename)

    def __repr__(self) -> str:
        """Return canonical string representation."""
        return f'{self.__class__.__name__}({self._key!r})'

    def __str__(self) -> str:
        """Return string representation."""
        return self.render(renderer='plaintext')

    def _repr_html_(self) -> str:
        """Represent using markdown renderer in a notebook environment."""
        return self.render(renderer='html')

    def render(self, renderer: str = 'html') -> str:
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
        formatter = pybtex.plugin.find_plugin('pybtex.style.formatting',
                                              style)()

        try:
            formatter = formatter.format_entry(self._key, self._entry)
            rendered = formatter.text.render(backend)
        except Exception as err:
            raise RenderError(
                f'Could not render {self._key!r}: {err}') from None

        return rendered
