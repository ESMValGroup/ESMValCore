"""API for handing recipe output."""

import base64
import logging
from collections.abc import Mapping
from pathlib import Path

import iris

from .config import Session
from .recipe_info import RecipeInfo
from .recipe_metadata import Contributor, Reference
from .templates import get_template

logger = logging.getLogger(__name__)


class TaskOutput:
    """Container for task output.

    Parameters
    ----------
    name : str
        Name of the task
    files : dict
        Mapping of the filenames with the associated attributes.
    """
    def __init__(self, name: str, files: dict):
        self.name = name
        self.files = tuple(
            OutputFile.create(filename, attributes)
            for filename, attributes in files.items())

    def __str__(self):
        """Return string representation."""
        return str(self.files)

    def __repr__(self):
        """Return canonical string representation."""
        indent = '  '
        string = f'{self.name}:\n'
        for file in self.files:
            string += f'{indent}{file}\n'
        return string

    def __len__(self):
        """Return number of files."""
        return len(self.files)

    def __getitem__(self, key: str):
        """Get item indexed by `key`."""
        return self.files[key]

    @property
    def image_files(self) -> tuple:
        """Return a tuple of image objects."""
        return tuple(item for item in self.files if item.kind == 'image')

    @property
    def data_files(self) -> tuple:
        """Return a tuple of data objects."""
        return tuple(item for item in self.files if item.kind == 'data')

    @classmethod
    def from_task(cls, task) -> 'TaskOutput':
        """Create an instance of `TaskOutput` from a Task.

        Where task is an instance of `esmvalcore._task.BaseTask`.
        """
        product_attributes = task.get_product_attributes()
        return cls(name=task.name, output=product_attributes)


class RecipeOutput(Mapping):
    """Container for recipe output.

    Parameters
    ----------
    task_output : dict
        Dictionary with recipe output grouped by task name. Each task value is
        a mapping of the filenames with the product attributes.
    """
    def __init__(self, task_output: dict, session=None, info=None):
        self._raw_task_output = task_output
        self._task_output = {}
        self.info = info
        self.session = session
        for task_name, files in task_output.items():
            self._task_output[task_name] = TaskOutput(name=task_name,
                                                      files=files)

    def __repr__(self):
        """Return canonical string representation."""
        string = '\n'.join(repr(item) for item in self._task_output.values())

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

    @classmethod
    def from_core_recipe_output(cls, recipe_output: dict):
        """Construct instance from `_recipe.Recipe` output.

        The core recipe format is not directly compatible with the API. This
        constructor does the following:

        1. Convert `config-user` dict to an instance of :obj:`Session`
        2. Converts the raw recipe dict to :obj:`RecipeInfo`

        Parameters
        ----------
        recipe_output : dict
            Output from `_recipe.Recipe.get_product_output`
        """
        task_output = recipe_output['task_output']
        recipe_data = recipe_output['recipe_data']
        recipe_config = recipe_output['recipe_config']
        recipe_filename = recipe_output['recipe_filename']

        session = Session.from_config_user(recipe_config)
        info = RecipeInfo(recipe_data, filename=recipe_filename)
        info.resolve()

        return cls(task_output, session=session, info=info)

    def write_html(self):
        """Write output summary to html document.

        A html file `index.html` gets written to the session directory.
        """
        filename = self.session.session_dir / 'index.html'

        template = get_template('recipe_output_page.j2')
        html_dump = self.render(template=template)

        with open(filename, 'w') as file:
            file.write(html_dump)

        logger.info("Wrote recipe output to:\nfile://%s", filename)

    def render(self, template=None):
        """Render output as html.

        template : :obj:`Template`
            An instance of :py:class:`jinja2.Template` can be passed to
            customize the output.
        """
        if not template:
            template = get_template(self.__class__.__name__ + '.j2')
        rendered = template.render(
            tasks=self.values(),
            session=self.session,
            info=self.info,
        )

        return rendered

    def read_main_log(self) -> str:
        """Read log file."""
        return self.session.main_log.read_text()

    def read_main_log_debug(self) -> str:
        """Read debug log file."""
        return self.session.main_log_debug.read_text()


class OutputFile():
    """Base container for recipe output files.

    Use `OutputFile.create(path='<path>', attributes=attributes)` to
    initialize a suitable subclass.

    Parameters
    ----------
    path : str
        Name of output file
    attributes : dict
        Attributes corresponding to the recipe output
    """

    kind = None

    def __init__(self, path: str, attributes: dict = None):
        if not attributes:
            attributes = {}

        self.attributes = attributes
        self.path = Path(path)

        self._authors = None
        self._references = None

    def __repr__(self):
        """Return canonical string representation."""
        return f'{self.__class__.__name__}({self.path.name!r})'

    @property
    def caption(self) -> str:
        """Return the caption of the file (fallback to path)."""
        return self.attributes.get('caption', str(self.path))

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

    def _get_derived_path(self, append: str, suffix: str = None):
        """Return path of related files.

        Parameters
        ----------
        append : str
            Add this string to the stem of the path.
        suffix : str
            The file extension to use (i.e. `.txt`)

        Returns
        -------
        Path
        """
        if not suffix:
            suffix = self.path.suffix
        return self.path.with_name(self.path.stem + append + suffix)

    @property
    def citation_file(self):
        """Return path of citation file (bibtex format)."""
        return self._get_derived_path('_citation', '.bibtex')

    @property
    def data_citation_file(self):
        """Return path of data citation info (txt format)."""
        return self._get_derived_path('_data_citation_info', '.txt')

    @property
    def provenance_svg_file(self):
        """Return path of provenance file (svg format)."""
        return self._get_derived_path('_provenance', '.svg')

    @property
    def provenance_xml_file(self):
        """Return path of provenance file (xml format)."""
        return self._get_derived_path('_provenance', '.xml')

    @classmethod
    def create(cls, path: str, attributes: dict = None):
        """Construct new instances of OutputFile.

        Chooses a derived class if suitable.
        """
        ext = Path(path).suffix
        if ext in ('.png', ):
            item_class = ImageFile
        elif ext in ('.nc', ):
            item_class = DataFile
        else:
            item_class = cls

        return item_class(path=path, attributes=attributes)


class ImageFile(OutputFile):
    """Container for image output."""

    kind = 'image'

    def to_base64(self) -> str:
        """Encode image as base64 to embed in a Jupyter notebook."""
        with open(self.path, "rb") as file:
            encoded = base64.b64encode(file.read())
        return encoded.decode('utf-8')

    def _repr_html_(self):
        """Render png as html in Jupyter notebook."""
        html_image = self.to_base64()
        return f"{self.caption}<img src='data:image/png;base64,{html_image}'/>"


class DataFile(OutputFile):
    """Container for data output."""

    kind = 'data'

    def load_xarray(self):
        """Load data using xarray."""
        # local import because `ESMValCore` does not depend on `xarray`
        import xarray as xr
        return xr.load_dataset(self.path)

    def load_iris(self):
        """Load data using iris."""
        return iris.load(str(self.path))
