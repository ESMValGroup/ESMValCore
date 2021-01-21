"""API for handing recipe output."""

from pathlib import Path
from pprint import pformat

from .recipe_metadata import Contributor, Reference


class OutputItem():
    """Base container for recipe output.

    Use `OutputItem.create(path='<filename>')` to initialize a suitable
    subclass.

    Parameters
    ----------
    filename : str
        Name of output file
    attributes : dict
        Attributes corresponding to the recipe output
    """

    kind = None

    def __init__(self, filename: str, attributes: dict = None):
        if not attributes:
            attributes = {}

        self.attributes = attributes
        self.filename = Path(filename)

        self._authors = None
        self._references = None

    def __repr__(self):
        """Return canonical string representation."""
        return (f'{self.__class__.__name__}(filename={self.filename.name!r},'
                f'\nattributes={pformat(self.attributes)})')

    def __str__(self):
        """Return string representation."""
        return f'{self.__class__.__name__}(filename={self.filename.name!r})'

    @property
    def authors(self) -> tuple:
        """List of recipe authors."""
        if self._authors is None:
            authors = self.attributes['authors']
            self._authors = tuple(
                Contributor.from_dict(author) for author in authors)
        return self._authors

    @property
    def references(self) -> tuple:
        """List of project references."""
        if self._references is None:
            tags = self.attributes.get('references', [])
            self._references = tuple(Reference.from_tag(tag) for tag in tags)
        return self._references

    def _get_derived_filename(self, append: str, suffix: str = None):
        """Return filename of related files.

        Parameters
        ----------
        append : str
            Add this string to the stem of the filename.
        suffix : str
            The file extension to use (i.e. `.txt`)

        Returns
        -------
        Path
        """
        if not suffix:
            suffix = self.filename.suffix
        return self.filename.with_name(self.filename.stem + append + suffix)

    @property
    def citation_file(self):
        """Return path of citation file (bibtex format)."""
        return self._get_derived_filename('_citation', '.bibtex')

    @property
    def data_citation_file(self):
        """Return path of data citation info (txt format)."""
        return self._get_derived_filename('_data_citation_info', '.txt')

    @property
    def provenance_svg_file(self):
        """Return path of provenance file (svg format)."""
        return self._get_derived_filename('_provenance', '.svg')

    @property
    def provenance_xml_file(self):
        """Return path of provenance file (xml format)."""
        return self._get_derived_filename('_provenance', '.xml')

    @classmethod
    def create(cls, filename: str, attributes: dict = None):
        """Constructor for new instances of OutputItem.

        Chooses a derived class if suitable.
        """
        ext = Path(filename).suffix
        if ext in ('.png', ):
            item_class = ImageItem
        elif ext in ('.nc', ):
            item_class = DataItem
        else:
            item_class = cls

        return item_class(filename=filename, attributes=attributes)


class ImageItem(OutputItem):
    """Container for image output."""

    kind = 'image'

    def _repr_html_(self):
        """Render png as html in Jupyter notebook."""
        import base64
        with open(self.filename, "rb") as f:
            encoded = base64.b64encode(f.read())
        html_string = encoded.decode('utf-8')
        caption = self.attributes['caption']
        return f"{caption}<img src='data:image/png;base64,{html_string}'/>"


class DataItem(OutputItem):
    """Container for data output."""

    kind = 'data'

    def load_xarray(self):
        """Load data using xarray."""
        import xarray as xr
        return xr.load_dataset(self.filename)

    def load_iris(self):
        """Load data using iris."""
        import iris
        return iris.load(str(self.filename))
