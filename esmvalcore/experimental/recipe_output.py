"""API for handing recipe output."""
import base64
import getpass
import logging
import os.path
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, Tuple, Type

import iris

from ..config._config import TASKSEP
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
        self.title = name.replace('_', ' ').replace(TASKSEP, ': ').title()
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

    def __getitem__(self, index: int):
        """Get item indexed by `index`."""
        return self.files[index]

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
        return cls(name=task.name, files=product_attributes)


class DiagnosticOutput:
    """Container for diagnostic output.

    Parameters
    ----------
    name : str
        Name of the diagnostic
    title: str
        Title of the diagnostic
    description: str
        Description of the diagnostic
    task_output : :obj:`list` of :obj:`TaskOutput`
        List of task output.
    """

    def __init__(self, name, task_output, title=None, description=None):
        self.name = name
        self.title = title if title else name.title()
        self.description = description if description else ''
        self.task_output = task_output

    def __repr__(self):
        """Return canonical string representation."""
        indent = '  '
        string = f'{self.name}:\n'
        for task_output in self.task_output:
            string += f'{indent}{task_output}\n'
        return string


class RecipeOutput(Mapping):
    """Container for recipe output.

    Parameters
    ----------
    task_output : dict
        Dictionary with recipe output grouped by task name. Each task value is
        a mapping of the filenames with the product attributes.

    Attributes
    ----------
    diagnostics : dict
        Dictionary with recipe output grouped by diagnostic.
    info : RecipeInfo
        The recipe used to create the output.
    session : esmvalcore.config.Session
        The session used to run the recipe.
    """

    FILTER_ATTRS: list = [
        "realms",
        "plot_type",  # Used by several diagnostics
        "plot_types",
        "long_names",
    ]

    def __init__(self, task_output: dict, session=None, info=None):
        self._raw_task_output = task_output
        self._task_output = {}
        self.diagnostics = {}
        self.info = info
        self.session = session

        # Group create task output and group by diagnostic
        diagnostics: dict = {}
        for task_name, files in task_output.items():
            name = task_name.split(TASKSEP)[0]
            if name not in diagnostics:
                diagnostics[name] = []
            task = TaskOutput(name=task_name, files=files)
            self._task_output[task_name] = task
            diagnostics[name].append(task)

        # Create diagnostic output
        filters: dict = {}
        for name, tasks in diagnostics.items():
            diagnostic_info = info.data['diagnostics'][name]
            self.diagnostics[name] = DiagnosticOutput(
                name=name,
                task_output=tasks,
                title=diagnostic_info.get('title'),
                description=diagnostic_info.get('description'),
            )

            # Add data to filters
            for task in tasks:
                for file in task.files:
                    RecipeOutput._add_to_filters(filters, file.attributes)

        # Sort at the end because sets are unordered
        self.filters = RecipeOutput._sort_filters(filters)

    @classmethod
    def _add_to_filters(cls, filters, attributes):
        """Add valid values to the HTML output filters."""
        for attr in RecipeOutput.FILTER_ATTRS:
            if attr not in attributes:
                continue
            values = attributes[attr]
            # `set()` to avoid duplicates
            attr_list = filters.get(attr, set())
            if (isinstance(values, str) or not isinstance(values, Sequence)):
                attr_list.add(values)
            else:
                attr_list.update(values)
            filters[attr] = attr_list

    @classmethod
    def _sort_filters(cls, filters):
        """Sort the HTML output filters."""
        for _filter, _attrs in filters.items():
            filters[_filter] = sorted(_attrs)
        return filters

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
        constructor converts the raw recipe dict to :obj:`RecipeInfo`

        Parameters
        ----------
        recipe_output : dict
            Output from `_recipe.Recipe.get_product_output`
        """
        task_output = recipe_output['task_output']
        recipe_data = recipe_output['recipe_data']
        session = recipe_output['session']
        recipe_filename = recipe_output['recipe_filename']

        info = RecipeInfo(recipe_data, filename=recipe_filename)
        info.resolve()

        return cls(task_output, session=session, info=info)

    def _log_ssh_html_info(self):
        """Log information about accessing index.html on an SSH server."""
        if 'SSH_CONNECTION' not in os.environ:
            return
        server_ip = os.environ['SSH_CONNECTION'].split()[2]
        server_ip_env = '${server}'
        server = f'{getpass.getuser()}@{server_ip_env}'
        port = '31415'
        port_env = '${port}'
        command = (
            f'server={server_ip} && port={port} && '
            f'ssh -t -L {port_env}:localhost:{port_env} {server} '
            f'{sys.executable} -m http.server {port_env} -d '
            f'{self.session.session_dir}'
        )
        logger.info(
            "It looks like you are connected to a remote machine via SSH. To "
            "show the output html file, you can try the following command on "
            "your local machine:\n%s\nThen visit http://localhost:%s in your "
            "browser",
            command,
            port,
        )
        logger.info(
            "If the port %s is already in use, you can replace it with any "
            "other free one (e.g., 12789). If you are connected through a "
            "jump host, replace the server IP address %s with your SSH server "
            "name",
            port,
            server_ip,
        )

    def write_html(self):
        """Write output summary to html document.

        A html file `index.html` gets written to the session directory.
        """
        filename = self.session.session_dir / 'index.html'

        template = get_template('recipe_output_page.j2')
        html_dump = self.render(template=template)

        with open(filename, 'w', encoding='utf-8') as file:
            file.write(html_dump)

        logger.info("Wrote recipe output to:\nfile://%s", filename)
        self._log_ssh_html_info()

    def render(self, template=None):
        """Render output as html.

        template : :obj:`Template`
            An instance of :py:class:`jinja2.Template` can be passed to
            customize the output.
        """
        if not template:
            template = get_template(self.__class__.__name__ + '.j2')
        rendered = template.render(
            diagnostics=self.diagnostics.values(),
            session=self.session,
            info=self.info,
            filters=self.filters,
            relpath=os.path.relpath,
        )

        return rendered

    def read_main_log(self) -> str:
        """Read log file."""
        return self.session.main_log.read_text(encoding='utf-8')

    def read_main_log_debug(self) -> str:
        """Read debug log file."""
        return self.session.main_log_debug.read_text(encoding='utf-8')


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

    kind: Optional[str] = None

    def __init__(self, path: str, attributes: Optional[dict] = None):
        if not attributes:
            attributes = {}

        self.attributes = attributes
        self.path = Path(path)

        self._authors: Optional[Tuple[Contributor, ...]] = None
        self._references: Optional[Tuple[Reference, ...]] = None

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

    def _get_derived_path(self, append: str, suffix: Optional[str] = None):
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
    def provenance_xml_file(self):
        """Return path of provenance file (xml format)."""
        return self._get_derived_path('_provenance', '.xml')

    @classmethod
    def create(
        cls,
        path: str,
        attributes: Optional[dict] = None,
    ) -> 'OutputFile':
        """Construct new instances of OutputFile.

        Chooses a derived class if suitable.
        """
        item_class: Type[OutputFile]

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
