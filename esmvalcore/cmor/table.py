"""CMOR information reader for ESMValTool.

Read variable information from CMOR 2 and CMOR 3 tables and make it
easily available for the other components of ESMValTool
"""

from __future__ import annotations

import copy
import glob
import importlib
import json
import logging
import os
from collections import Counter
from functools import lru_cache, total_ordering
from pathlib import Path
from typing import TYPE_CHECKING, Self

import yaml

from esmvalcore.exceptions import RecipeError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from io import TextIOWrapper

    from esmvalcore.config import Config, Session
    from esmvalcore.typing import Facets

logger = logging.getLogger(__name__)

CMOR_TABLES: dict[str, InfoBase] = {}
"""dict of str, obj: CMOR info objects.

.. note::
    If this dictionary is empty, it can be populated by loading the global
    configuration by importing the :mod:`esmvalcore.config` module.
"""

_CMOR_KEYS = (
    "standard_name",
    "long_name",
    "units",
    "modeling_realm",
    "frequency",
)


def _get_institutes(project: str, dataset: str) -> list[str]:
    """Return the institutes from the controlled vocabulary given the dataset name."""
    try:
        return CMOR_TABLES[project].institutes[dataset]  # type: ignore[attr-defined]
    except (KeyError, AttributeError):
        return []


def _get_activity(
    project: str,
    exp: str | list[str],
) -> str | list[str] | None:
    """Return the activity from the controlled vocabulary given the experiment name."""
    try:
        if isinstance(exp, list):
            return [CMOR_TABLES[project].activities[value][0] for value in exp]  # type: ignore[attr-defined]
        return CMOR_TABLES[project].activities[exp][0]  # type: ignore[attr-defined]
    except (KeyError, AttributeError):
        return None


def _update_cmor_facets(facets: Facets) -> None:
    """Update `facets` with information from CMOR table."""
    project: str = facets["project"]  # type: ignore[assignment]
    mip: str = facets["mip"]  # type: ignore[assignment]
    short_name: str = facets["short_name"]  # type: ignore[assignment]
    derive: bool = facets.get("derive", False)  # type: ignore[assignment]
    table = CMOR_TABLES.get(project)
    if table:
        table_entry = table.get_variable(
            mip,
            short_name,
            branding_suffix=facets.get("branding_suffix"),  # type: ignore[arg-type]
            derived=derive,
        )
    else:
        table_entry = None
    if table_entry is None:
        msg = (
            f"Unable to load CMOR table (project) '{project}' for variable "
            f"'{short_name}' with mip '{mip}'"
        )
        raise RecipeError(msg)
    facets["original_short_name"] = table_entry.short_name
    for key in _CMOR_KEYS:
        if key not in facets:
            value = getattr(table_entry, key, None)
            if value is not None:
                facets[key] = value
            else:
                logger.debug(
                    "Failed to add key %s to variable %s from CMOR table",
                    key,
                    facets,
                )
    if "dataset" in facets and "institute" not in facets:
        institute = _get_institutes(project, facets["dataset"])  # type: ignore[arg-type]
        if institute:
            facets["institute"] = institute
    if "exp" in facets and "activity" not in facets:
        activity = _get_activity(project, facets["exp"])  # type: ignore[arg-type]
        if activity:
            facets["activity"] = activity


def _get_mips(project: str, short_name: str) -> list[str]:
    """Get all available MIP tables in a project."""
    tables = CMOR_TABLES[project].tables
    return [
        mip
        for mip, table in tables.items()
        if short_name in table
        or any(short_name == vardef.short_name for vardef in table.values())
    ]


def _get_branding_suffixes(
    project: str,
    mip: str,
    short_name: str,
) -> list[str]:
    """Get all available branding suffixes for a variable in a MIP table."""
    table = CMOR_TABLES[project].tables[mip]
    return [
        branded_name.split("_", 1)[1]
        for branded_name, vardef in table.items()
        if short_name == vardef.short_name and "_" in branded_name
    ]


def get_var_info(
    project: str,
    mip: str,
    short_name: str,
    branding_suffix: str | None = None,
) -> VariableInfo | None:
    """Get variable information.

    Note
    ----
    If `project=CORDEX` and the `mip` ends with 'hr', it is cropped to 'h'
    since CORDEX X-hourly tables define the `mip` as ending in 'h' instead of
    'hr'.

    Parameters
    ----------
    project:
        Dataset's project.
    mip:
        Variable's CMOR table, i.e., MIP.
    short_name:
        Variable's short name.
    branding_suffix:
        A suffix that will be appended to ``short_name`` when looking up the
        variable in the CMOR table.

    Returns
    -------
    VariableInfo | None
        `VariableInfo` object for the requested variable if found, ``None``
        otherwise.

    Raises
    ------
    KeyError
        No CMOR tables available for `project`.

    """
    if project not in CMOR_TABLES:
        msg = (
            f"No CMOR tables available for project '{project}'. The following "
            f"tables are available: {', '.join(CMOR_TABLES)}."
        )
        raise KeyError(msg)

    # CORDEX X-hourly tables define the mip as ending in 'h' instead of 'hr'
    if project == "CORDEX" and mip.endswith("hr"):
        mip = mip.replace("hr", "h")

    return CMOR_TABLES[project].get_variable(
        mip,
        short_name,
        branding_suffix=branding_suffix,
    )


def read_cmor_tables(cfg_developer: Path | None = None) -> None:
    """Read cmor tables required in the configuration.

    .. deprecated:: 2.14.0

        The config-developer.yml file based configuration is deprecated and
        will no longer be supported in ESMValCore v2.16.0. Please use
        :func:`~esmvalcore.cmor.table.load_cmor_tables` instead of this function.

    Parameters
    ----------
    cfg_developer:
        Path to config-developer.yml file.

    Raises
    ------
    TypeError
        If `cfg_developer` is not a Path-like object
    """
    if cfg_developer is None:
        cfg_developer = Path(__file__).parents[1] / "config-developer.yml"
    elif not isinstance(cfg_developer, Path):
        msg = "cfg_developer is not a Path-like object, got "
        raise TypeError(msg, cfg_developer)
    mtime = cfg_developer.stat().st_mtime
    cmor_tables = _read_cmor_tables(cfg_developer, mtime)
    CMOR_TABLES.clear()
    CMOR_TABLES.update(cmor_tables)


@lru_cache
def _read_cmor_tables(
    cfg_file: Path,
    mtime: float,  # noqa: ARG001
) -> dict[str, InfoBase]:
    """Read cmor tables required in the configuration.

    Parameters
    ----------
    cfg_file: pathlib.Path
        Path to config-developer.yml file.
    mtime: float
        Modification time of config-developer.yml file. Only used by the
        `lru_cache` decorator to make sure the file is read again when it
        is changed.
    """
    with cfg_file.open("r", encoding="utf-8") as file:
        cfg_developer = yaml.safe_load(file)
    cwd = os.path.dirname(os.path.realpath(__file__))
    var_alt_names_file = os.path.join(cwd, "variable_alt_names.yml")
    with open(var_alt_names_file, encoding="utf-8") as yfile:
        alt_names = yaml.safe_load(yfile)

    cmor_tables: dict[str, InfoBase] = {}

    # Try to infer location for custom tables from config-developer.yml file,
    # if not possible, use default location
    custom_path = None
    if "custom" in cfg_developer:
        custom_path = cfg_developer["custom"].get("cmor_path")
    if custom_path is not None:
        custom_path = os.path.expandvars(os.path.expanduser(custom_path))
    custom = CustomInfo(custom_path)
    cmor_tables["custom"] = custom

    install_dir = os.path.dirname(os.path.realpath(__file__))
    for table in cfg_developer:
        if table == "custom":
            continue
        cmor_tables[table] = _read_table(
            cfg_developer,
            table,
            install_dir,
            custom,
            alt_names,
        )
    return cmor_tables


def _read_table(cfg_developer, table, install_dir, custom, alt_names):
    project = cfg_developer[table]
    cmor_type = project.get("cmor_type", "CMIP5")
    default_path = os.path.join(install_dir, "tables", cmor_type.lower())
    table_path = project.get("cmor_path", default_path)
    table_path = os.path.expandvars(os.path.expanduser(table_path))
    cmor_strict = project.get("cmor_strict", True)
    default_table_prefix = project.get("cmor_default_table_prefix", "")

    if cmor_type == "CMIP3":
        return CMIP3Info(
            table_path,
            default=custom,
            strict=cmor_strict,
            alt_names=alt_names,
        )

    if cmor_type == "CMIP5":
        return CMIP5Info(
            table_path,
            default=custom,
            strict=cmor_strict,
            alt_names=alt_names,
        )

    if cmor_type == "CMIP6":
        return CMIP6Info(
            table_path,
            default=custom,
            strict=cmor_strict,
            default_table_prefix=default_table_prefix,
            alt_names=alt_names,
        )
    msg = f"Unsupported CMOR type {cmor_type}"
    raise ValueError(msg)


_TABLE_CACHE: dict[str, InfoBase] = {}
"""The CMOR tables are cached for faster access."""


def clear_table_cache() -> None:
    """Clear the CMOR table cache."""
    _TABLE_CACHE.clear()


def get_tables(
    session: Session | Config,
    project: str,
) -> InfoBase:
    """Get the CMOR tables for a project.

    Parameters
    ----------
    session:
        The configuration.
    project:
        The project to load a CMOR table for.
    """
    if project not in session["projects"]:
        msg = f"Unknown project '{project}', please configure it under 'projects'."
        raise ValueError(msg)

    kwargs = (
        session["projects"][project]
        .get(
            "cmor_table",
            {
                "type": "esmvalcore.cmor.table.NoInfo",
            },
        )
        .copy()
    )
    if "type" not in kwargs:
        msg = (
            f"Missing CMOR table 'type' in configuration of project {project}. "
            f"Current configuration is:\n{yaml.safe_dump(kwargs)}"
        )
        raise ValueError(msg)
    cache_key = str(kwargs)
    if cache_key not in _TABLE_CACHE:
        module_name, cls_name = kwargs.pop("type").rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        tables = cls(**kwargs)
        if not isinstance(tables, InfoBase):
            msg = (
                "`type` should be a subclass `esmvalcore.cmor.table.InfoBase`, "
                f"but your configuration for project '{project}' contains "
                f"'{tables}' of type: '{type(tables)}'."
            )
            raise TypeError(msg)
        _TABLE_CACHE[cache_key] = tables

    return _TABLE_CACHE[cache_key]


class InfoBase:
    """Base class for all CMOR table info classes.

    Parameters
    ----------
    default:
        Default table to look variables on if not found.

        .. deprecated:: 2.14.0

            The ``default`` parameter is deprecated and will be removed in
            ESMValCore v2.16.0. Please use the ``paths`` parameter instead
            to aggregate multiple tables.

    alt_names:
        List of known alternative names for variables. If no value is provided,
        the default values from the installed copy of
        `variable_alt_names.yml <https://github.com/ESMValGroup/ESMValCore/blob/main/esmvalcore/cmor/variable_alt_names.yml>`_
        will be used.

    strict:
        If :obj:`False`, the function :meth:`~esmvalcore.cmor.table.InfoBase.get_variable`
        will look for a variable in other tables if it can not be found in the
        table specified by ``mip`` in the :ref:`recipe <recipe>` or :class:`~esmvalcore.dataset.Dataset`.

    paths:
        A list of paths to CMOR tables. The path can be relative to the built-in
        tables in the
        `esmvalcore/cmor/tables <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables>`_
        directory, or any other path. The built-in tables will be used if the
        path is relative and exists in the built-in tables directory.
    """

    def __init__(
        self,
        default: CustomInfo | None = None,
        alt_names: list[list[str]] | None = None,
        strict: bool = True,
        paths: Iterable[Path] = (),
    ) -> None:
        # Configure the paths to the CMOR tables.
        builtin_tables_path = Path(__file__).parent / "tables"
        paths = tuple(Path(os.path.expandvars(p)).expanduser() for p in paths)
        self.paths = tuple(
            builtin_tables_path / p
            if (builtin_tables_path / p).is_dir()
            else p
            for p in paths
        )
        """A list of paths to CMOR tables."""
        for path in self.paths:
            if not path.is_dir():
                raise NotADirectoryError(path)

        # Configure the alternative names.
        if alt_names is None:
            alt_names_path = Path(__file__).parent / "variable_alt_names.yml"
            alt_names = yaml.safe_load(
                alt_names_path.read_text(encoding="utf-8"),
            )
        self.alt_names = alt_names
        """List of known alternative names for variables."""
        self.coords: dict[str, CoordinateInfo] = {}
        """The coordinates defined in these tables."""
        self.default = default
        """
        Default table to look variables on if not found.

        .. deprecated:: 2.14.0

            The ``default`` attribute is deprecated and will be removed in
            ESMValCore v2.16.0.
        """
        self.strict = strict
        """If False, will look for a variable in other tables if it can not be
        found in the requested one.
        """
        self.tables: dict[str, TableInfo] = {}
        """A mapping from table names to :class:`TableInfo` objects."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(paths={list(self.paths)}, strict={self.strict}, alt_names={self.alt_names})"

    def get_table(self, table: str) -> TableInfo | None:
        """Search and return the table info.

        Parameters
        ----------
        table: str
            Table name

        Returns
        -------
        TableInfo
            Return the TableInfo object for the requested table if
            found, returns None if not
        """
        return self.tables.get(table)

    def get_variable(
        self,
        table_name: str,
        short_name: str,
        *,
        branding_suffix: str | None = None,
        derived: bool = False,
    ) -> VariableInfo | None:
        """Search and return the variable information.

        Parameters
        ----------
        table_name:
            Table name, i.e., the ``mip`` in the :ref:`recipe <recipe>` or
            :class:`~esmvalcore.dataset.Dataset`.
        short_name:
            Variable's short name.
        branding_suffix:
            A suffix that will be appended to ``short_name`` when looking up the
            variable in the CMOR table.
        derived:
            Variable is derived. Information retrieval for derived variables
            always looks in the default tables (usually, the custom tables) if
            variable is not found in the requested table.

        Returns
        -------
        VariableInfo | None
            `VariableInfo` object for the requested variable if found, ``None``
            otherwise.

        """
        alt_names_list = self._get_alt_names_list(short_name)
        if branding_suffix:
            # The branding suffix was introduced for CMIP7. The branded variable
            # name used in the CMOR tables is the short_name followed by an
            # an underscore and the branding suffix.
            #
            # For projects prior to CMIP7 the name used in the CMOR table may
            # also contain a suffix, but without an underscore. For example
            # ch4Clim in the CMIP6 Amon table, where ch4 is the short_name and
            # Clim is the suffix. This is not a branding suffix, but we can
            # use it to select the correct variable in the CMOR table anyway.
            alt_names_list = [
                f"{name}_{branding_suffix}" for name in alt_names_list
            ] + [f"{name}{branding_suffix}" for name in alt_names_list]

        # First, look in requested table
        table = self.get_table(table_name)
        if table:
            for alt_names in alt_names_list:
                try:
                    return table[alt_names]
                except KeyError:
                    pass

        # If that didn't work, look in all tables (i.e., other MIPs) if
        # cmor_strict=False or derived=True
        var_info = self._look_in_all_tables(derived, alt_names_list)

        # If that didn't work either, look in default table if
        # cmor_strict=False or derived=True
        if not var_info and self.default is not None:
            var_info = self._look_in_default(
                derived,
                alt_names_list,
                table_name,
            )

        # If necessary, adapt frequency of variable (set it to the one from the
        # requested MIP). E.g., if the user asked for table `Amon`, but the
        # variable has been found in `day`, use frequency `mon`.
        if var_info:
            var_info = var_info.copy()
            var_info = self._update_frequency_from_mip(table_name, var_info)

        return var_info

    def _look_in_default(self, derived, alt_names_list, table_name):
        """Look for variable in default table."""
        # TODO: remove in v2.16.0
        var_info = None
        if not self.strict or derived:
            for alt_names in alt_names_list:
                var_info = self.default.get_variable(table_name, alt_names)
                if var_info:
                    break
        return var_info

    def _look_in_all_tables(self, derived, alt_names_list):
        """Look for variable in all tables."""
        var_info = None
        if not self.strict or derived:
            for alt_names in alt_names_list:
                var_info = self._look_all_tables(alt_names)
                if var_info:
                    break
        return var_info

    def _get_alt_names_list(self, short_name):
        """Get list of alternative variable names."""
        alt_names_list = [short_name]
        for alt_names in self.alt_names:
            if short_name in alt_names:
                alt_names_list.extend(
                    [
                        alt_name
                        for alt_name in alt_names
                        if alt_name not in alt_names_list
                    ],
                )
        return alt_names_list

    def _update_frequency_from_mip(self, table_name, var_info):
        """Update frequency information of var_info from table."""
        mip_info = self.get_table(table_name)
        if mip_info:
            var_info.frequency = mip_info.frequency
        return var_info

    def _look_all_tables(self, alt_names):
        """Look for variable in all tables."""
        for table_vars in sorted(self.tables.values()):
            if alt_names in table_vars:
                return table_vars[alt_names]
        return None


class CMIP6Info(InfoBase):
    """Class to read CMIP6-like CMOR tables.

    This class reads CMOR 3 json format tables.

    Parameters
    ----------
    cmor_tables_path:
        The path to a directory with subdirectory "Tables" where the CMOR tables
        are located.

        .. deprecated:: 2.14.0

            The ``cmor_tables_path`` parameter is deprecated and will be removed in
            ESMValCore v2.16.0. Please use the ``paths`` parameter instead.

    default:
        Default table to look variables on if not found.

        .. deprecated:: 2.14.0

            The ``default`` parameter is deprecated and will be removed in
            ESMValCore v2.16.0. Please use the ``paths`` parameter instead
            to aggregate multiple tables.

    alt_names:
        List of known alternative names for variables. If no value is provided,
        the default values from the installed copy of
        `variable_alt_names.yml`_ will be used.

    strict:
        If :obj:`False`, the function :meth:`~esmvalcore.cmor.table.InfoBase.get_variable`
        will look for a variable in other tables if it can not be found in the
        table specified by ``mip`` in the :ref:`recipe <recipe>` or
        :class:`~esmvalcore.dataset.Dataset`.

    default_table_prefix:
        If the table_id contains a prefix, it can be specified here.

        .. deprecated:: 2.14.0

            The ``default_table_prefix`` parameter is deprecated and will be removed in
            ESMValCore v2.16.0.

    paths:
        A list of paths to CMOR tables. The path can be relative to the built-in
        tables in the
        `esmvalcore/cmor/tables <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables>`_
        directory, or any other path. The built-in tables will be used if the
        path is relative and exists in the built-in tables directory.

    """

    def __init__(
        self,
        cmor_tables_path: str | None = None,
        default: CustomInfo | None = None,
        alt_names: list[list[str]] | None = None,
        strict: bool = True,
        default_table_prefix: str = "",
        paths: Iterable[Path] = (),
    ) -> None:
        if cmor_tables_path is not None:
            # Support cmor_tables_path for backward compatibility.
            # TODO: remove in v2.16.0
            tables_path = Path(self._get_cmor_path(cmor_tables_path))
            if (tables_path / "tables").exists():
                # Support CMIP7 which uses a lowercase "tables" subdirectory.
                cmor_folder = tables_path / "tables"
            else:
                cmor_folder = tables_path / "Tables"
            paths = (*tuple(paths), cmor_folder)
        super().__init__(default, alt_names, strict, paths=paths)

        self.default_table_prefix = default_table_prefix
        """
        If the table_id contains a prefix, it can be specified here.

        .. deprecated:: 2.14.0

            The ``default_table_prefix`` attribute is deprecated and will be
            removed in ESMValCore v2.16.0.
        """
        self.var_to_freq: dict[str, dict[str, str]] = {}
        self.activities: dict[str, list[str]] = {}
        """A mapping from ``exp`` to ``activity`` from the controlled vocabulary."""
        self.institutes: dict[str, list[str]] = {}
        """A mapping from ``dataset`` to ``institute`` from the controlled vocabulary."""

        for path in self.paths:
            if not any(path.glob("*.json")):
                msg = f"No CMOR tables found in {path}"
                raise ValueError(msg)
            self._load_controlled_vocabulary(path)
            self._load_coordinates(path)
            for json_file in glob.glob(os.path.join(path, "*.json")):
                if "CV_test" in json_file or "grids" in json_file:
                    continue
                try:
                    self._load_table(json_file)
                except Exception:
                    msg = f"Exception raised when loading {json_file}"
                    # Logger may not be ready at this stage
                    if logger.handlers:
                        logger.error(msg)
                    else:
                        print(msg)  # noqa: T201
                    raise

    @staticmethod
    def _get_cmor_path(cmor_tables_path: str) -> str:
        if os.path.isdir(cmor_tables_path):
            return cmor_tables_path
        cwd = os.path.dirname(os.path.realpath(__file__))
        cmor_tables_path = os.path.join(cwd, "tables", cmor_tables_path)
        if os.path.isdir(cmor_tables_path):
            return cmor_tables_path
        msg = f"CMOR tables not found in {cmor_tables_path}"
        raise ValueError(msg)

    def _load_table(self, json_file):
        with open(json_file, encoding="utf-8") as inf:
            raw_data = json.loads(inf.read())
            if not self._is_table(raw_data):
                return
            header = raw_data["Header"]
            table_name = header["table_id"].split(" ")[-1]
            if table_name not in self.tables:
                table = TableInfo()
                table.name = table_name
                self.tables[table_name] = table
            table = self.tables[table_name]

            generic_levels = header["generic_levels"].split()
            self.var_to_freq[table.name] = {}

            for var_name, var_data in raw_data["variable_entry"].items():
                var = VariableInfo("CMIP6")
                var.read_json(var_data, table.frequency)
                self._assign_dimensions(var, generic_levels)
                table[var_name] = var
                self.var_to_freq[table.name][var_name] = var.frequency

            if not table.frequency:
                var_freqs = (var.frequency for var in table.values())
                table_freq, _ = Counter(var_freqs).most_common(1)[0]
                table.frequency = table_freq

    def _assign_dimensions(self, var, generic_levels):
        for dimension in var.dimensions:
            if dimension in generic_levels:
                coord = CoordinateInfo(dimension)
                coord.generic_level = True
                for name in self.coords:
                    generic_level = self.coords[name].generic_lev_name
                    if dimension in [generic_level]:
                        coord.generic_lev_coords[name] = self.coords[name]
            else:
                try:
                    coord = self.coords[dimension]
                except KeyError:
                    logger.exception(
                        "Can not find dimension %s for variable %s",
                        dimension,
                        var,
                    )
                    raise

            var.coordinates[dimension] = coord

    def _load_coordinates(self, path: Path) -> None:
        for json_file in glob.glob(
            os.path.join(path, "*coordinate*.json"),
        ):
            with open(json_file, encoding="utf-8") as inf:
                table_data = json.loads(inf.read())
                for coord_name in table_data["axis_entry"]:
                    coord = CoordinateInfo(coord_name)
                    coord.read_json(table_data["axis_entry"][coord_name])
                    self.coords[coord_name] = coord

    def _load_controlled_vocabulary(self, path: Path) -> None:
        for json_file in glob.glob(
            os.path.join(path, "*_CV.json"),
        ):
            with open(json_file, encoding="utf-8") as inf:
                table_data = json.loads(inf.read())
                try:
                    exps = table_data["CV"]["experiment_id"]
                    for exp_id in exps:
                        activity = exps[exp_id]["activity_id"][0].split(" ")
                        self.activities[exp_id] = activity
                except (KeyError, AttributeError):
                    pass

                try:
                    sources = table_data["CV"]["source_id"]
                    for source_id in sources:
                        institution = sources[source_id]["institution_id"]
                        self.institutes[source_id] = institution
                except (KeyError, AttributeError):
                    pass

    def get_table(self, table: str) -> TableInfo | None:
        """Search and return the table info.

        Parameters
        ----------
        table:
            Table name

        Returns
        -------
        :
            Return the TableInfo object for the requested table if
            found, returns None if not
        """
        try:
            return self.tables[table]
        except KeyError:
            return self.tables.get(f"{self.default_table_prefix}{table}")

    @staticmethod
    def _is_table(table_data):
        if "variable_entry" not in table_data:
            return False
        return "Header" in table_data


class Obs4MIPsInfo(CMIP6Info):
    """Class to read obs4MIPs-like CMOR tables.

    Parameters
    ----------
    alt_names:
        List of known alternative names for variables. If no value is provided,
        the default values from the installed copy of
        `variable_alt_names.yml`_ will be used.

    strict:
        If :obj:`False`, the function :meth:`~esmvalcore.cmor.table.InfoBase.get_variable`
        will look for a variable in other tables if it can not be found in the
        table specified by ``mip`` in the :ref:`recipe <recipe>` or
        :class:`~esmvalcore.dataset.Dataset`.

    paths:
        A list of paths to CMOR tables. The path can be relative to the built-in
        tables in the
        `esmvalcore/cmor/tables <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables>`_
        directory, or any other path. The built-in tables will be used if the
        path is relative and exists in the built-in tables directory.
    """

    def __init__(
        self,
        alt_names: list[list[str]] | None = None,
        strict: bool = True,
        paths: Iterable[Path] = (),
    ) -> None:
        super().__init__(
            alt_names=alt_names,
            strict=strict,
            paths=paths,
        )
        # Remove the prefix from the table_id.
        table_id_prefix = "obs4MIPs_"
        for name in list(self.tables):
            if name.startswith(table_id_prefix):
                table = self.tables.pop(name)
                self.tables[name[len(table_id_prefix) :]] = table


@total_ordering
class TableInfo(dict):
    """Container class for storing a CMOR table."""

    def __init__(self, *args, **kwargs):
        """Create a new TableInfo object for storing VariableInfo objects."""
        super().__init__(*args, **kwargs)
        self.name = ""
        """Table name."""
        self.frequency = ""
        """Table frequency (if defined)."""
        self.realm = ""
        """Table realm (if defined)."""

    def __eq__(self, other):
        return (self.name, self.frequency, self.realm) == (
            other.name,
            other.frequency,
            other.realm,
        )

    def __ne__(self, other):
        return (self.name, self.frequency, self.realm) != (
            other.name,
            other.frequency,
            other.realm,
        )

    def __lt__(self, other):
        return (self.name, self.frequency, self.realm) < (
            other.name,
            other.frequency,
            other.realm,
        )


class JsonInfo:
    """Base class for the info classes.

    Provides common utility methods to read json variables
    """

    def __init__(self):
        self._json_data = {}

    def _read_json_variable(self, parameter, default=""):
        """Read a json parameter in json_data.

        Parameters
        ----------
        parameter: str
            parameter to read

        Returns
        -------
        str
            Option's value or empty string if parameter is not present
        """
        if parameter not in self._json_data:
            return default
        return str(self._json_data[parameter])

    def _read_json_list_variable(self, parameter):
        """Read a json list parameter in json_data.

        Parameters
        ----------
        parameter: str
            parameter to read

        Returns
        -------
        list
            Option's value or empty list if parameter is not present
        """
        if parameter not in self._json_data:
            return []
        value = self._json_data[parameter]
        if isinstance(value, str):
            value = value.split()
        return value


class VariableInfo(JsonInfo):
    """Class to read and store variable information."""

    def __init__(
        self,
        table_type: str = "",
        short_name: str = "",
    ) -> None:
        """Class to read and store variable information.

        Parameters
        ----------
        table_type:
            Type of table (e.g., CMIP5, CMIP6).

            .. deprecated:: 2.14.0

                The ``table_type`` parameter is deprecated and will be removed
                in ESMValCore v2.16.0.
        short_name:
            Variable's short name.

            .. deprecated:: 2.14.0

                The ``short_name`` parameter is deprecated and will be removed
                in ESMValCore v2.16.0.
        """
        super().__init__()
        self.table_type = table_type
        self.modeling_realm: list[str] = []
        """Modeling realm"""
        self.short_name = short_name
        """Short name"""
        self.standard_name = ""
        """Standard name"""
        self.long_name = ""
        """Long name"""
        self.units = ""
        """Data units"""
        self.valid_min = ""
        """Minimum admitted value"""
        self.valid_max = ""
        """Maximum admitted value"""
        self.frequency = ""
        """Data frequency"""
        self.positive = ""
        """Increasing direction"""

        self.dimensions: list[str] = []
        """List of dimensions"""
        self.coordinates: dict[str, CoordinateInfo] = {}
        """Coordinates

        This is a dict with the names of the dimensions as keys and
        CoordinateInfo objects as values.
        """

        self._json_data = None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} defining variable '{self.short_name}'>"

    def copy(self) -> Self:
        """Return a shallow copy of VariableInfo.

        Returns
        -------
        VariableInfo
           Shallow copy of this object.
        """
        return copy.copy(self)

    def read_json(self, json_data: dict, default_freq: str) -> None:
        """Read variable information from json.

        Non-present options will be set to empty

        Parameters
        ----------
        json_data:
            Dictionary created by the json reader containing variable
            information.
        default_freq:
            Default frequency to use if it is not defined at variable level.
        """
        self._json_data = json_data

        self.short_name = self._read_json_variable("out_name")
        self.standard_name = self._read_json_variable("standard_name")
        self.long_name = self._read_json_variable("long_name")
        self.units = self._read_json_variable("units")
        self.valid_min = self._read_json_variable("valid_min")
        self.valid_max = self._read_json_variable("valid_max")
        self.positive = self._read_json_variable("positive")
        self.modeling_realm = self._read_json_variable(
            "modeling_realm",
        ).split()
        self.frequency = self._read_json_variable("frequency", default_freq)

        # "dimensions" is a list of str in CMIP7 and a space separated str in CMIP6 CMOR tables.
        self.dimensions = self._read_json_list_variable("dimensions")

    def has_coord_with_standard_name(self, standard_name: str) -> bool:
        """Check if a coordinate with a given `standard_name` exists.

        For some coordinates, multiple (slightly different) versions with
        different dimension names but identical `standard_name` exist. For
        example, the CMIP6 tables provide 4 different `standard_name=time`
        dimensions: `time`, `time1`, `time2`, and `time3`. Other examples would
        be the CMIP6 pressure levels (`plev19`, `plev23`, `plev27`, etc.  with
        standard name `air_pressure`) and the altitudes (`alt16`, `alt40` with
        standard name `altitude`).

        This function can be used to check for the existence of a specific
        coordinate defined by its `standard_name`, not its dimension name.

        Parameters
        ----------
        standard_name:
            Standard name to be checked.

        Returns
        -------
        :
            `True` if there is at least one coordinate with the given
            `standard_name`, `False` if not.

        """
        for coord in self.coordinates.values():
            if coord.standard_name == standard_name:
                return True
        return False


class CoordinateInfo(JsonInfo):
    """Class to read and store coordinate information."""

    def __init__(self, name: str) -> None:
        """Class to read and store coordinate information.

        Parameters
        ----------
        name:
            coordinate's name
        """
        super().__init__()
        self.name = name
        """Name of the coordinate entry in the CMOR table."""
        self.generic_level = False
        self.generic_lev_coords: dict[str, CoordinateInfo] = {}

        self.axis = ""
        """Axis"""
        self.value = ""
        """Coordinate value"""
        self.standard_name = ""
        """Standard name"""
        self.long_name = ""
        """Long name"""
        self.out_name = ""
        """
        Out name

        This is the name of the variable in the file
        """
        self.var_name = ""
        """Short name"""
        self.units = ""
        """Units"""
        self.stored_direction = ""
        """Direction in which the coordinate increases"""
        self.requested: list[str] = []
        """Values requested"""
        self.valid_min = ""
        """Minimum allowed value"""
        self.valid_max = ""
        """Maximum allowed value"""
        self.must_have_bounds = ""
        """Whether bounds are required on this dimension"""
        self.generic_lev_name = ""
        """Generic level name"""

    def read_json(self, json_data):
        """Read coordinate information from json.

        Non-present options will be set to empty

        Parameters
        ----------
        json_data: dict
            dictionary created by the json reader containing
            coordinate information
        """
        self._json_data = json_data

        self.axis = self._read_json_variable("axis")
        self.value = self._read_json_variable("value")
        self.out_name = self._read_json_variable("out_name")
        self.var_name = self._read_json_variable("out_name")
        self.standard_name = self._read_json_variable("standard_name")
        self.long_name = self._read_json_variable("long_name")
        self.units = self._read_json_variable("units")
        self.stored_direction = self._read_json_variable("stored_direction")
        self.valid_min = self._read_json_variable("valid_min")
        self.valid_max = self._read_json_variable("valid_max")
        self.requested = self._read_json_list_variable("requested")
        self.must_have_bounds = self._read_json_variable("must_have_bounds")
        self.generic_lev_name = self._read_json_variable("generic_level_name")


class CMIP5Info(InfoBase):
    """Class to read CMIP5-like CMOR tables.

    This class reads CMOR 2 format tables.

    Parameters
    ----------
    cmor_tables_path:
        The path to a directory with subdirectory "Tables" where the CMOR tables
        are located.

        .. deprecated:: 2.14.0

            The ``cmor_tables_path`` parameter is deprecated and will be removed in
            ESMValCore v2.16.0. Please use the ``paths`` parameter instead.

    default:
        Default table to look variables on if not found.

        .. deprecated:: 2.14.0

            The ``default`` parameter is deprecated and will be removed in
            ESMValCore v2.16.0. Please use the ``paths`` parameter instead
            to aggregate multiple tables.

    alt_names:
        List of known alternative names for variables. If no value is provided,
        the default values from the installed copy of
        `variable_alt_names.yml`_ will be used.

    strict:
        If :obj:`False`, the function :meth:`~esmvalcore.cmor.table.InfoBase.get_variable`
        will look for a variable in other tables if it can not be found in the
        table specified by ``mip`` in the :ref:`recipe <recipe>` or
        :class:`~esmvalcore.dataset.Dataset`.

    default_table_prefix:
        If the table_id contains a prefix, it can be specified here.

        .. deprecated:: 2.14.0

            The ``default_table_prefix`` parameter is deprecated and will be removed in
            ESMValCore v2.16.0.

    paths:
        A list of paths to CMOR tables. The path can be relative to the built-in
        tables in the
        `esmvalcore/cmor/tables <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables>`_
        directory, or any other path. The built-in tables will be used if the
        path is relative and exists in the built-in tables directory.

    """

    def __init__(
        self,
        cmor_tables_path: str | None = None,
        default: CustomInfo | None = None,
        alt_names: list[list[str]] | None = None,
        strict: bool = True,
        paths: Iterable[Path] = (),
    ) -> None:
        if cmor_tables_path is not None:
            # Support cmor_tables_path for backward compatibility.
            # TODO: remove in v2.16.0
            cmor_tables_path = self._get_cmor_path(cmor_tables_path)
            cmor_folder = Path(cmor_tables_path) / "Tables"
            paths = (*tuple(paths), cmor_folder)
        super().__init__(default, alt_names, strict, paths=paths)

        self._current_table: TextIOWrapper | None = None
        self._last_line_read = ("", "")

        for path in self.paths:
            for table_file in sorted(
                glob.glob(os.path.join(path, "*")),
                # Read coordinate files before variable files so we can link the
                # variables with the coordinates.
                key=lambda filename: "coordinate" not in filename,
            ):
                if "_grids" in table_file:
                    continue
                try:
                    self._load_table(table_file)
                except Exception:
                    msg = f"Exception raised when loading {table_file}"
                    # Logger may not be ready at this stage
                    if logger.handlers:
                        logger.error(msg)
                    else:
                        print(msg)  # noqa: T201
                    raise

    @staticmethod
    def _get_cmor_path(cmor_tables_path):
        if os.path.isdir(cmor_tables_path):
            return cmor_tables_path
        cwd = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(cwd, "tables", cmor_tables_path)

    def _load_table(self, table_file: str) -> None:
        table = self._read_table_file(table_file)
        if table.name in self.tables:
            self.tables[table.name].update(table)
        else:
            self.tables[table.name] = table

    def _read_table_file(self, table_file: str) -> TableInfo:
        table = TableInfo()
        with open(table_file, encoding="utf-8") as self._current_table:
            self._read_line()
            while True:
                key, value = self._last_line_read
                if key == "table_id":
                    table.name = value[len("Table ") :]
                elif key == "frequency":
                    table.frequency = value
                elif key == "modeling_realm":
                    table.realm = value
                elif key == "generic_levels":
                    for dim in value.split(" "):
                        coord = CoordinateInfo(dim)
                        coord.generic_level = True
                        coord.axis = "Z"
                        self.coords[dim] = coord
                elif key == "axis_entry":
                    self.coords[value] = self._read_coordinate(value)
                    continue
                elif key == "variable_entry":
                    table[value] = self._read_variable(value, table.frequency)
                    continue
                if not self._read_line():
                    break
        return table

    def _read_line(self):
        line = self._current_table.readline()
        if line == "":
            return False
        if line.startswith("!"):
            return self._read_line()
        line = line.replace("\n", "")
        if "!" in line:
            line = line[: line.index("!")]
        line = line.strip()
        if not line:
            self._last_line_read = ("", "")
        else:
            index = line.index(":")
            self._last_line_read = (
                line[:index].strip(),
                line[index + 1 :].strip(),
            )
        return True

    def _read_coordinate(self, value):
        coord = CoordinateInfo(value)
        while self._read_line():
            key, value = self._last_line_read
            if key in ("variable_entry", "axis_entry"):
                return coord
            if key == "requested":
                coord.requested.extend(val for val in value.split(" ") if val)
                continue
            if hasattr(coord, key):
                setattr(coord, key, value)
        return coord

    def _read_variable(self, entry_name, frequency):
        var = VariableInfo(table_type="CMIP5")
        var.frequency = frequency
        while self._read_line():
            key, value = self._last_line_read
            if key in ("variable_entry", "axis_entry"):
                break
            if key in ("dimensions", "modeling_realm"):
                setattr(var, key, value.split())
            elif hasattr(var, key):
                setattr(var, key, value)
            elif key == "out_name":
                var.short_name = value
        if not var.short_name:
            # Some of our custom CMIP5 table entries are missing the `out_name` field.
            # In that case, we assume the entry name is the same as short_name.
            var.short_name = entry_name
        for dim in var.dimensions:
            var.coordinates[dim] = self.coords[dim]
        return var

    def get_table(self, table: str) -> TableInfo | None:
        """Search and return the table info.

        Parameters
        ----------
        table:
            Table name

        Returns
        -------
        :
            Return the TableInfo object for the requested table if
            found, returns None if not
        """
        return self.tables.get(table)


class CMIP3Info(CMIP5Info):
    """Class to read CMIP3-like CMOR tables.

    Parameters
    ----------
    cmor_tables_path:
        The path to a directory with subdirectory "Tables" where the CMOR tables
        are located.

        .. deprecated:: 2.14.0

            The ``cmor_tables_path`` parameter is deprecated and will be removed in
            ESMValCore v2.16.0. Please use the ``paths`` parameter instead.

    default:
        Default table to look variables on if not found.

        .. deprecated:: 2.14.0

            The ``default`` parameter is deprecated and will be removed in
            ESMValCore v2.16.0. Please use the ``paths`` parameter instead
            to aggregate multiple tables.

    alt_names:
        List of known alternative names for variables. If no value is provided,
        the default values from the installed copy of
        `variable_alt_names.yml`_ will be used.

    strict:
        If :obj:`False`, the function :meth:`~esmvalcore.cmor.table.InfoBase.get_variable`
        will look for a variable in other tables if it can not be found in the
        table specified by ``mip`` in the :ref:`recipe <recipe>` or
        :class:`~esmvalcore.dataset.Dataset`.

    default_table_prefix:
        If the table_id contains a prefix, it can be specified here.

        .. deprecated:: 2.14.0

            The ``default_table_prefix`` parameter is deprecated and will be removed in
            ESMValCore v2.16.0.

    paths:
        A list of paths to CMOR tables. The path can be relative to the built-in
        tables in the
        `esmvalcore/cmor/tables <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables>`_
        directory, or any other path. The built-in tables will be used if the
        path is relative and exists in the built-in tables directory.

    """

    def _read_table_file(self, table_file: str) -> TableInfo:
        for dim in ("zlevel",):
            coord = CoordinateInfo(dim)
            coord.generic_level = True
            coord.axis = "Z"
            self.coords[dim] = coord
        return super()._read_table_file(table_file)

    def _read_coordinate(self, value):
        coord = super()._read_coordinate(value)
        if not coord.out_name:
            coord.out_name = coord.name
            coord.var_name = coord.name
        return coord

    def _read_variable(self, entry_name, frequency):
        var = super()._read_variable(entry_name, frequency)
        var.frequency = ""
        var.modeling_realm = []
        return var


class CustomInfo(CMIP5Info):
    """Class to read custom var info for ESMVal.

    .. deprecated:: 2.14.0

        This class is deprecated and will be removed in ESMValCore v2.16.0.
        Please use :class:`~esmvalcore.cmor.tables.table.CMIP5Info` instead.

    Parameters
    ----------
    cmor_tables_path:
        Full path to the table or name for the table if it is present in
        ESMValTool repository. If ``None``, use default tables from
        `esmvalcore/cmor/tables/custom`.

    """

    def __init__(self, cmor_tables_path: str | Path | None = None) -> None:
        """Initialize class member."""
        self.coords = {}
        self.tables = {}
        self.var_to_freq: dict[str, dict[str, str]] = {}
        table = TableInfo()
        table.name = "custom"
        self.tables[table.name] = table

        # First, read default custom tables from repository
        self.paths = (Path(self._get_cmor_path("cmip5-custom")),)

        # Second, if given, update default tables with user-defined custom
        # tables
        if cmor_tables_path is not None:
            user_table_folder = Path(self._get_cmor_path(cmor_tables_path))
            if not user_table_folder.is_dir():
                msg = (
                    f"Custom CMOR tables path {user_table_folder} is "
                    f"not a directory"
                )
                raise ValueError(msg)
            self.paths += (user_table_folder,)

        for path in self.paths:
            self._read_table_dir(str(path))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(paths={list(self.paths)})"

    def _read_table_dir(self, table_dir: str) -> None:
        """Read CMOR tables from directory."""
        # If present, read coordinates
        coordinates_file = os.path.join(table_dir, "CMOR_coordinates.dat")
        if os.path.isfile(coordinates_file):
            self._read_table_file(coordinates_file)

        # Read other variables
        for dat_file in glob.glob(os.path.join(table_dir, "*.dat")):
            if dat_file == coordinates_file:
                continue
            try:
                self._load_table(dat_file)
            except Exception:
                msg = f"Exception raised when loading {dat_file}"
                # Logger may not be ready at this stage
                if logger.handlers:
                    logger.error(msg)
                else:
                    print(msg)  # noqa: T201
                raise

    def get_variable(
        self,
        table_name: str,  # noqa: ARG002
        short_name: str,
        *,
        branding_suffix: str | None = None,  # noqa: ARG002
        derived: bool = False,  # noqa: ARG002
    ) -> VariableInfo | None:
        """Search and return the variable info.

        Parameters
        ----------
        table:
            Table name. Ignored for custom tables.
        short_name:
            Variable's short name.
        branding_suffix:
            A suffix that will be appended to ``short_name`` when looking up the
            variable in the CMOR table. Ignored for custom tables.
        derived:
            Variable is derived. Info retrieval for derived variables always
            looks on the default tables if variable is not found in the
            requested table. Ignored for custom tables.

        Returns
        -------
        VariableInfo | None
            `VariableInfo` object for the requested variable if found, returns
            None if not.

        """
        return self.tables["custom"].get(short_name, None)

    def _read_table_file(self, table_file: str) -> TableInfo:
        """Read a single table file."""
        table = TableInfo()
        table.name = "custom"
        with open(table_file, encoding="utf-8") as self._current_table:
            self._read_line()
            while True:
                key, value = self._last_line_read
                if key == "generic_levels":
                    for dim in value.split(" "):
                        coord = CoordinateInfo(dim)
                        coord.generic_level = True
                        coord.axis = "Z"
                        self.coords[dim] = coord
                elif key == "axis_entry":
                    self.coords[value] = self._read_coordinate(value)
                    continue
                elif key == "variable_entry":
                    table[value] = self._read_variable(value, "")
                    continue
                if not self._read_line():
                    return table


class NoInfo(InfoBase):
    """Table that can be used for projects that do not provide a CMOR table."""

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def get_variable(
        self,
        table_name: str,  # noqa: ARG002
        short_name: str,
        *,
        branding_suffix: str | None = None,  # noqa: ARG002
        derived: bool = False,  # noqa: ARG002
    ) -> VariableInfo | None:
        """Search and return the variable information.

        Parameters
        ----------
        table_name:
            Table name, i.e., the ``mip`` in the :ref:`recipe <recipe>` or
            :class:`~esmvalcore.dataset.Dataset`.
        short_name:
            Variable's short name.
        branding_suffix:
            A suffix that will be appended to ``short_name`` when looking up the
            variable in the CMOR table.
        derived:
            Variable is derived. Information retrieval for derived variables
            always looks in the default tables (usually, the custom tables) if
            variable is not found in the requested table.

        Returns
        -------
        VariableInfo | None
            `VariableInfo` object for the requested variable if found, ``None``
            otherwise.

        """
        vardef = VariableInfo()
        vardef.short_name = short_name
        return vardef
