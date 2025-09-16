from pathlib import Path

import numpy as np
import pytest
import yaml

import esmvalcore
from esmvalcore import __version__ as current_version
from esmvalcore.config import CFG
from esmvalcore.config._config_validators import (
    ValidationError,
    _handle_deprecation,
    _listify_validator,
    deprecate_extra_facets_dir,
    validate_bool,
    validate_bool_or_none,
    validate_check_level,
    validate_config_developer,
    validate_diagnostics,
    validate_float,
    validate_int,
    validate_int_or_none,
    validate_int_positive_or_none,
    validate_path,
    validate_path_or_none,
    validate_positive,
    validate_projects,
    validate_rootpath,
    validate_search_esgf,
    validate_string,
    validate_string_or_none,
)
from esmvalcore.exceptions import (
    ESMValCoreDeprecationWarning,
    InvalidConfigParameter,
)


def generate_validator_testcases(valid):
    # The code for this function was taken from matplotlib (v3.3) and modified
    # to fit the needs of ESMValCore. Matplotlib is licenced under the terms of
    # the the 'Python Software Foundation License'
    # (https://www.python.org/psf/license)

    validation_tests = (
        {
            "validator": validate_bool,
            "success": ((True, True), (False, False)),
            "fail": ("fail", 2, -1, []),
        },
        {
            "validator": validate_check_level,
            "success": (
                (1, 1),
                (5, 5),
                ("dEBUG", 1),
                ("default", 3),
            ),
            "fail": (6, 0, "fail"),
        },
        {
            "validator": validate_diagnostics,
            "success": (
                ("/", {"/"}),
                ("a ", {"a/*"}),
                ("/ a ", {"/", "a/*"}),
                ("/ a a", {"/", "a/*"}),
                (("/", "a"), {"/", "a/*"}),
                ([], set()),
            ),
            "fail": (1, [1, 2]),
        },
        {
            "validator": _listify_validator(validate_float, n_items=2),
            "success": (
                (_, [1.5, 2.5])
                for _ in (
                    "1.5, 2.5",
                    [1.5, 2.5],
                    [1.5, 2.5],
                    (1.5, 2.5),
                    np.array((1.5, 2.5)),
                )
            ),
            "fail": ("fail", ("a", 1), (1, 2, 3)),
        },
        {
            "validator": _listify_validator(validate_float, n_items=2),
            "success": (
                (_, [1.5, 2.5])
                for _ in (
                    "1.5, 2.5",
                    [1.5, 2.5],
                    [1.5, 2.5],
                    (1.5, 2.5),
                    np.array((1.5, 2.5)),
                )
            ),
            "fail": ("fail", ("a", 1), (1, 2, 3)),
        },
        {
            "validator": _listify_validator(validate_int, n_items=2),
            "success": (
                (_, [1, 2])
                for _ in ("1, 2", [1.5, 2.5], [1, 2], (1, 2), np.array((1, 2)))
            ),
            "fail": ("fail", ("a", 1), (1, 2, 3)),
        },
        {
            "validator": validate_bool_or_none,
            "success": ((None, None), (True, True), (False, False)),
            "fail": ("A", 1),
        },
        {
            "validator": validate_int_or_none,
            "success": ((None, None),),
            "fail": (),
        },
        {
            "validator": validate_int_positive_or_none,
            "success": ((None, None),),
            "fail": (),
        },
        {
            "validator": validate_path,
            "success": (
                ("a/b/c", Path.cwd() / "a" / "b" / "c"),
                ("/a/b/c/", Path("/", "a", "b", "c")),
                ("~/", Path.home()),
                (Path.home(), Path.home()),
            ),
            "fail": (None, 123, False, []),
        },
        {
            "validator": validate_path_or_none,
            "success": (
                ("a/b/c", Path.cwd() / "a" / "b" / "c"),
                ("/a/b/c/", Path("/", "a", "b", "c")),
                ("~/", Path.home()),
                (None, None),
            ),
            "fail": (123, False, []),
        },
        {
            "validator": validate_rootpath,
            "success": (
                # Test a single path
                ({"default": "/a"}, {"default": [Path("/a")]}),
                ({"default": Path("/a")}, {"default": [Path("/a")]}),
                # Test a list of paths
                ({"CMIP6": ["/a", "/b"]}, {"CMIP6": [Path("/a"), Path("/b")]}),
                (
                    {"CMIP6": [Path("/a"), Path("/b")]},
                    {"CMIP6": [Path("/a"), Path("/b")]},
                ),
                # Test a dict of paths
                (
                    {
                        "CMIP6": {
                            "/a": "DKRZ",
                            "/b": "ESGF",
                        },
                    },
                    {
                        "CMIP6": {
                            "/a": "DKRZ",
                            "/b": "ESGF",
                        },
                    },
                ),
            ),
            "fail": (),
        },
        {
            "validator": validate_positive,
            "success": (
                (0.1, 0.1),
                (1, 1),
                (1.5, 1.5),
            ),
            "fail": (0, -1, "fail"),
        },
        {
            "validator": _listify_validator(validate_string),
            "success": (
                ("", []),
                ("a,b", ["a", "b"]),
                ("abc", ["abc"]),
                ("abc, ", ["abc"]),
                ("abc, ,", ["abc"]),
                (["a", "b"], ["a", "b"]),
                (("a", "b"), ["a", "b"]),
                (iter(["a", "b"]), ["a", "b"]),
                (np.array(["a", "b"]), ["a", "b"]),
                ((1, 2), ["1", "2"]),
                (np.array([1, 2]), ["1", "2"]),
            ),
            "fail": (set(), 1),
        },
        {
            "validator": validate_string_or_none,
            "success": ((None, None),),
            "fail": (),
        },
        {
            "validator": validate_search_esgf,
            "success": (
                ("never", "never"),
                ("NEVER", "never"),
                ("when_missing", "when_missing"),
                ("WhEN_MIssIng", "when_missing"),
                ("always", "always"),
                ("Always", "always"),
            ),
            "fail": (0, 3.14, True, "fail"),
        },
        {
            "validator": validate_projects,
            "success": (
                ({"CMIP6": {}}, {"CMIP6": {}}),
                (
                    {"CMIP6": {"extra_facets": {}}},
                    {"CMIP6": {"extra_facets": {}}},
                ),
            ),
            "fail": (
                0,
                3.14,
                True,
                "fail",
                {"CMIP6": {"extra_facets": 0}},
                {"CMIP6": {"invalid_option": 0}},
            ),
        },
    )

    for validator_dict in validation_tests:
        validator = validator_dict["validator"]
        if valid:
            for arg, target in validator_dict["success"]:
                yield validator, arg, target
        else:
            for arg in validator_dict["fail"]:
                yield validator, arg


@pytest.mark.parametrize(
    ("validator", "arg", "target"),
    generate_validator_testcases(True),
)
def test_validator_valid(validator, arg, target):
    res = validator(arg)
    assert res == target


@pytest.mark.parametrize(
    ("validator", "arg"),
    generate_validator_testcases(False),
)
def test_validator_invalid(validator, arg):
    with pytest.raises(ValidationError):
        validator(arg)


@pytest.mark.parametrize("remove_version", [current_version, "0.0.1", "9.9.9"])
def test_handle_deprecation(remove_version):
    """Test ``_handle_deprecation``."""
    option = "test_var"
    deprecated_version = "2.7.0"
    more_info = " More information on this is not available."

    if remove_version != "9.9.9":
        msg = (
            r"The configuration option or command line argument `test_var` "
            r"has been removed in ESMValCore version .* More information on "
            r"this is not available."
        )
        with pytest.raises(InvalidConfigParameter, match=msg):
            _handle_deprecation(
                option,
                deprecated_version,
                remove_version,
                more_info,
            )
    else:
        msg = (
            r"The configuration option or command line argument `test_var` "
            r"has been deprecated in ESMValCore version .* More information "
            r"on this is not available."
        )
        with pytest.warns(ESMValCoreDeprecationWarning, match=msg):
            _handle_deprecation(
                option,
                deprecated_version,
                remove_version,
                more_info,
            )


def test_validate_config_developer_none():
    """Test ``validate_config_developer``."""
    path = validate_config_developer(None)
    assert path == Path(esmvalcore.__file__).parent / "config-developer.yml"


def test_validate_config_developer(tmp_path):
    """Test ``validate_config_developer``."""
    custom_table_path = (
        Path(esmvalcore.__file__).parent / "cmor" / "tables" / "custom"
    )
    cfg_dev = {
        "custom": {"cmor_path": custom_table_path},
        "CMIP3": {"input_dir": {"default": "/"}},
        "CMIP5": {"input_dir": {"default": "/"}},
        "CMIP6": {"input_dir": {"default": "/"}},
        "CORDEX": {"input_dir": {"default": "/"}},
        "OBS": {"input_dir": {"default": "/"}},
        "OBS6": {"input_dir": {"default": "/"}},
        "obs4MIPs": {"input_dir": {"default": "/"}},
        "ana4mips": {"input_dir": {"default": "/"}},
        "native6": {"input_dir": {"default": "/"}},
        "EMAC": {"input_dir": {"default": "/"}},
        "IPSLCM": {"input_dir": {"default": "/"}},
        "ICON": {"input_dir": {"default": "/"}},
        "CESM": {"input_dir": {"default": "/"}},
    }
    cfg_dev_file = tmp_path / "cfg-developer.yml"
    with open(cfg_dev_file, mode="w", encoding="utf-8") as file:
        yaml.safe_dump(cfg_dev, file)

    path = validate_config_developer(cfg_dev_file)
    assert path == cfg_dev_file

    # Restore original config-developer file
    validate_config_developer(None)


# TODO: remove in v2.14.0
def test_extra_facets_dir_tuple_deprecated(monkeypatch):
    """Test extra_facets_dir."""
    with pytest.warns(ESMValCoreDeprecationWarning):
        monkeypatch.setitem(CFG, "extra_facets_dir", ("/extra/facets",))
    assert CFG["extra_facets_dir"] == [Path("/extra/facets")]


# TODO: remove in v2.15.0
def test_deprecate_extra_facets_dir(monkeypatch):
    """Test deprecate_extra_facets_dir."""
    monkeypatch.setenv("ESMVALTOOL_USE_NEW_EXTRA_FACETS_CONFIG", "1")
    msg = (
        r"Since the environment variable "
        r"ESMVALTOOL_USE_NEW_EXTRA_FACETS_CONFIG is set"
    )
    with pytest.warns(ESMValCoreDeprecationWarning, match=msg):
        deprecate_extra_facets_dir({}, "value", "validated_value")
