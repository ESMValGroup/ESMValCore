"""API for handing recipe output."""

import base64
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path

from .recipe_metadata import Contributor, Reference


class TaskOutput(Sequence):
    """Container for task output."""

    def __init__(self, output_items):
        self._output_files = tuple(
            OutputFile.create(filename, attributes)
            for filename, attributes in output_items.items())

    def __repr__(self):
        """Return canonical string representation."""
        return '\n'.join(str(item) for item in self._output_files)

    def __len__(self):
        """Return number of output objects."""
        return len(self._output_files)

    def __getitem__(self, key: str):
        """Get item indexed by `key`."""
        return self._output_files[key]

    @property
    def image_files(self) -> tuple:
        """Return a tuple of image objects."""
        return tuple(item for item in self._output_files
                     if item.kind == 'image')

    @property
    def data_files(self) -> tuple:
        """Return a tuple of data objects."""
        return tuple(item for item in self._output_files
                     if item.kind == 'data')


class RecipeOutput(Mapping):
    """Container for recipe output."""

    def __init__(self, raw_output):
        self._raw_output = raw_output
        self._task_output = {}
        for task, product_output in raw_output.items():
            self._task_output[task] = TaskOutput(product_output)

    def __repr__(self):
        """Return canonical string representation."""
        string = ''
        for key, value in self._task_output.items():
            string += f'{key}:\n'
            string += textwrap.indent(str(value), prefix=' ')
            string += '\n'

        return string

    def __getitem__(self, key: str):
        """Get task indexed by `key`."""
        return self._task_output[key]

    def __iter__(self):
        """Iterate over tasks."""
        yield from self._task_output

    def __len__(self):
        """Return number of tasks."""
        return len(self._task_output)


class OutputFile():
    """Base container for recipe output files.

    Use `OutputFile.create(path='<filename>')` to initialize a suitable
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
        return f'{self.__class__.__name__}({self.filename.name!r})'

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
        """Construct new instances of OutputFile.

        Chooses a derived class if suitable.
        """
        ext = Path(filename).suffix
        if ext in ('.png', ):
            item_class = ImageFile
        elif ext in ('.nc', ):
            item_class = DataFile
        else:
            item_class = cls

        return item_class(filename=filename, attributes=attributes)


class ImageFile(OutputFile):
    """Container for image output."""

    kind = 'image'

    def _repr_html_(self):
        """Render png as html in Jupyter notebook."""
        with open(self.filename, "rb") as file:
            encoded = base64.b64encode(file.read())

        html_string = encoded.decode('utf-8')
        caption = self.attributes['caption']
        return f"{caption}<img src='data:image/png;base64,{html_string}'/>"


class DataFile(OutputFile):
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
