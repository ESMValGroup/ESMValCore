from collections.abc import MutableMapping
from pathlib import Path
from textwrap import dedent

import pytest

import esmvalcore
import esmvalcore.cmor.table
import esmvalcore.config._config_object
from esmvalcore.config import CFG, Config, Session
from esmvalcore.config._config_object import DEFAULT_CONFIG_DIR
from esmvalcore.exceptions import (
    InvalidConfigParameter,
)


def test_config_class():
    config = {
        "log_level": "info",
        "exit_on_warning": False,
        "output_file_type": "png",
        "output_dir": "./esmvaltool_output",
        "auxiliary_data_dir": "./auxiliary_data",
        "save_intermediary_cubes": False,
        "remove_preproc_dir": True,
        "max_parallel_tasks": None,
        "profile_diagnostic": False,
        "rootpath": {"CMIP6": "~/data/CMIP6"},
        "drs": {"CMIP6": "default"},
    }

    cfg = Config(config)

    assert isinstance(cfg["output_dir"], Path)
    assert isinstance(cfg["auxiliary_data_dir"], Path)

    assert CFG


def test_config_update():
    config = Config({"output_dir": "directory"})
    fail_dict = {"output_dir": 123}

    with pytest.raises(InvalidConfigParameter):
        config.update(fail_dict)


@pytest.mark.parametrize("update_format", ["mapping", "kwargs", "tuple"])
def test_config_update_config_developer_set_last(
    monkeypatch: pytest.MonkeyPatch,
    update_format: str,
) -> None:
    monkeypatch.setattr(esmvalcore.cmor.table, "CMOR_TABLES", {})
    new_config = {
        "config_developer_file": Path(esmvalcore.__file__).parent
        / "config-developer.yml",
        "projects": {
            "CMIP6": {
                "cmor_table": {
                    "type": "esmvalcore.cmor.table.NoInfo",
                },
            },
        },
    }
    config = Config({"output_dir": "directory"})
    if update_format == "mapping":
        config.update(new_config)
    elif update_format == "kwargs":
        config.update(**new_config)
    elif update_format == "tuple":
        config.update(tuple(new_config.items()))

    assert len(esmvalcore.cmor.table.CMOR_TABLES) > 1
    assert "CMIP6" in esmvalcore.cmor.table.CMOR_TABLES
    assert isinstance(
        esmvalcore.cmor.table.CMOR_TABLES["CMIP6"],
        esmvalcore.cmor.table.CMIP6Info,
    )


def test_config_update_too_many_args() -> None:
    config = Config({"output_dir": "directory"})
    with pytest.raises(
        TypeError,
        match=r"Expected at most 1 positional argument, got 2",
    ):
        config.update(1, 2)


def test_set_bad_item():
    config = Config({"output_dir": "config"})
    with pytest.raises(InvalidConfigParameter) as err_exc:
        config["bad_item"] = 47

    assert str(err_exc.value) == "`bad_item` is not a valid config parameter."


def test_config_init():
    config = Config()
    assert isinstance(config, MutableMapping)


def test_config_key_error():
    config = Config()
    with pytest.raises(KeyError):
        config["invalid_key"]


def test_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cfg_default: Config,
) -> None:
    """Test `Config.reload`."""
    monkeypatch.setattr(
        esmvalcore.config._config_object,
        "USER_CONFIG_DIR",
        tmp_path,
    )
    cfg = Config()

    cfg.reload()

    assert cfg == cfg_default


def test_reload_fail(monkeypatch, tmp_path):
    """Test `Config.reload`."""
    monkeypatch.setenv("ESMVALTOOL_CONFIG_DIR", tmp_path)

    config_file = tmp_path / "invalid_config_file.yml"
    config_file.write_text("invalid_option: 1")
    monkeypatch.setattr(
        esmvalcore.config._config_object,
        "USER_CONFIG_DIR",
        tmp_path,
    )
    cfg = Config()

    with pytest.raises(InvalidConfigParameter):
        cfg.reload()


def test_session():
    config = Config({"output_dir": "config"})

    session = config.start_session("recipe_name")
    assert session == config

    session["output_dir"] = "session"
    assert session != config


def test_session_repr_small(session: Session) -> None:
    # See https://github.com/ESMValGroup/ESMValCore/issues/2868
    assert len(repr(session)) < 100


def test_session_str_small(session: Session) -> None:
    assert len(str(session)) < 100


def test_session_key_error():
    session = Session({})
    with pytest.raises(KeyError):
        session["invalid_key"]


def test_get_user_config_dir_and_source_with_env(tmp_path, monkeypatch):
    """Test `_get_user_config_dir` and `_get_user_config_source`."""
    monkeypatch.setenv("ESMVALTOOL_CONFIG_DIR", str(tmp_path))

    config_dir = esmvalcore.config._config_object._get_user_config_dir()
    config_src = esmvalcore.config._config_object._get_user_config_source()

    assert config_dir == tmp_path
    assert config_src == "ESMVALTOOL_CONFIG_DIR environment variable"


def test_get_user_config_dir_and_source_no_env(tmp_path, monkeypatch):
    """Test `_get_user_config_dir` and `_get_user_config_source`."""
    monkeypatch.delenv("ESMVALTOOL_CONFIG_DIR", raising=False)

    config_dir = esmvalcore.config._config_object._get_user_config_dir()
    config_src = esmvalcore.config._config_object._get_user_config_source()

    assert config_dir == Path("~/.config/esmvaltool").expanduser()
    assert config_src == "default user configuration directory"


def test_get_user_config_dir_with_env_fail(tmp_path, monkeypatch):
    """Test `_get_user_config_dir` and `_get_user_config_source`."""
    empty_path = tmp_path / "this" / "does" / "not" / "exist"
    monkeypatch.setenv("ESMVALTOOL_CONFIG_DIR", str(empty_path))

    msg = (
        "Invalid configuration directory specified via ESMVALTOOL_CONFIG_DIR "
        "environment variable:"
    )
    with pytest.raises(NotADirectoryError, match=msg):
        esmvalcore.config._config_object._get_user_config_dir()


def _setup_config_dirs(tmp_path):
    """Set up test configuration directories."""
    config1 = tmp_path / "config1" / "1.yml"
    config2a = tmp_path / "config2" / "2a.yml"
    config2b = tmp_path / "config2" / "2b.yml"
    config1.parent.mkdir(parents=True, exist_ok=True)
    config2a.parent.mkdir(parents=True, exist_ok=True)
    config1.write_text(
        dedent(
            """
        output_file_type: '1'
        rootpath:
          default: '1'
          '1': '1'
        """,
        ),
    )
    config2a.write_text(
        dedent(
            """
        output_file_type: '2a'
        rootpath:
          default: '2a'
          '2': '2a'
        """,
        ),
    )
    config2b.write_text(
        dedent(
            """
        output_file_type: '2b'
        rootpath:
          default: '2b'
          '2': '2b'
        """,
        ),
    )


@pytest.mark.parametrize(
    ("dirs", "output_file_type", "rootpath"),
    [
        ([], "png", {"default": "~/climate_data"}),
        (["/this/path/does/not/exist"], "png", {"default": "~/climate_data"}),
        (["{tmp_path}/config1"], "1", {"default": "1", "1": "1"}),
        (
            ["{tmp_path}/config1", "/this/path/does/not/exist"],
            "1",
            {"default": "1", "1": "1"},
        ),
        (
            ["{tmp_path}/config1", "{tmp_path}/config2"],
            "2b",
            {"default": "2b", "1": "1", "2": "2b"},
        ),
        (
            ["{tmp_path}/config2", "{tmp_path}/config1"],
            "1",
            {"default": "1", "1": "1", "2": "2b"},
        ),
    ],
)
def test_load_from_dirs(dirs, output_file_type, rootpath, tmp_path):
    """Test `Config.load_from_dirs`."""
    _setup_config_dirs(tmp_path)

    config_dirs = []
    for dir_ in dirs:
        config_dirs.append(dir_.format(tmp_path=str(tmp_path)))
    for name, path in rootpath.items():
        abspath = Path(path).expanduser().absolute()
        rootpath[name] = [abspath]

    cfg = Config()
    assert not cfg
    cfg["rootpath"] = {"X": "x"}
    cfg["search_data"] = "complete"

    cfg.load_from_dirs(config_dirs)

    assert cfg["output_file_type"] == output_file_type
    if any(Path(d).exists() for d in config_dirs):
        # Legacy setting "rootpath" is not available in default config.
        assert cfg["rootpath"] == rootpath
    assert cfg["search_data"] == "quick"


@pytest.mark.parametrize(
    ("cli_config_dir", "output"),
    [
        (None, [DEFAULT_CONFIG_DIR, "~/.config/esmvaltool"]),
        (Path("/c"), [DEFAULT_CONFIG_DIR, "~/.config/esmvaltool", "/c"]),
    ],
)
def test_get_all_config_dirs(cli_config_dir, output, monkeypatch):
    """Test `_get_all_config_dirs`."""
    monkeypatch.delenv("ESMVALTOOL_CONFIG_DIR", raising=False)
    excepted = []
    for out in output:
        excepted.append(Path(out).expanduser().absolute())

    config_dirs = esmvalcore.config._config_object._get_all_config_dirs(
        cli_config_dir,
    )

    assert config_dirs == excepted


@pytest.mark.parametrize(
    ("cli_config_dir", "output"),
    [
        (None, ["defaults", "default user configuration directory"]),
        (
            Path("/c"),
            [
                "defaults",
                "default user configuration directory",
                "command line argument",
            ],
        ),
    ],
)
def test_get_all_config_sources(cli_config_dir, output, monkeypatch):
    """Test `_get_all_config_sources`."""
    monkeypatch.delenv("ESMVALTOOL_CONFIG_DIR", raising=False)
    config_srcs = esmvalcore.config._config_object._get_all_config_sources(
        cli_config_dir,
    )
    assert config_srcs == output


@pytest.mark.parametrize(
    ("dirs", "output_file_type", "rootpath"),
    [
        ([], None, {"X": "x"}),
        (["/this/path/does/not/exist"], None, {"X": "x"}),
        (["{tmp_path}/config1"], "1", {"default": "1", "1": "1", "X": "x"}),
        (
            ["{tmp_path}/config1", "/this/path/does/not/exist"],
            "1",
            {"default": "1", "1": "1", "X": "x"},
        ),
        (
            ["{tmp_path}/config1", "{tmp_path}/config2"],
            "2b",
            {"default": "2b", "1": "1", "2": "2b", "X": "x"},
        ),
        (
            ["{tmp_path}/config2", "{tmp_path}/config1"],
            "1",
            {"default": "1", "1": "1", "2": "2b", "X": "x"},
        ),
    ],
)
def test_update_from_dirs(dirs, output_file_type, rootpath, tmp_path):
    """Test `Config.update_from_dirs`."""
    _setup_config_dirs(tmp_path)

    config_dirs = []
    for dir_ in dirs:
        config_dirs.append(dir_.format(tmp_path=str(tmp_path)))
    for name, path in rootpath.items():
        abspath = Path(path).expanduser().absolute()
        rootpath[name] = [abspath]

    cfg = Config()
    assert not cfg
    cfg["rootpath"] = {"X": "x"}
    cfg["search_data"] = "quick"

    cfg.update_from_dirs(config_dirs)

    if output_file_type is None:
        assert "output_file_type" not in cfg
    else:
        assert cfg["output_file_type"] == output_file_type
    assert cfg["rootpath"] == rootpath
    assert cfg["search_data"] == "quick"


def test_nested_update():
    """Test `Config.update_from_dirs`."""
    cfg = Config()
    assert not cfg

    cfg["drs"] = {"X": "x", "Z": "z"}
    cfg["search_data"] = "quick"

    cfg.nested_update({"drs": {"Y": "y", "X": "xx"}, "max_years": 1})

    assert len(cfg) == 3
    assert cfg["drs"] == {"Y": "y", "X": "xx", "Z": "z"}
    assert cfg["search_data"] == "quick"
    assert cfg["max_years"] == 1


def test_context_mapping():
    cfg = Config()
    assert not cfg
    with cfg.context({"output_dir": "/path/to/output"}):
        assert len(cfg) == 1
        assert cfg["output_dir"] == Path("/path/to/output")
    assert not cfg


def test_context_kwargs():
    cfg = Config()
    assert not cfg
    with cfg.context(output_dir="/path/to/output"):
        assert len(cfg) == 1
        assert cfg["output_dir"] == Path("/path/to/output")
    assert not cfg


def test_context_mapping_and_kwargs():
    cfg = Config()
    assert not cfg
    with cfg.context({"output_dir": "/o"}, auxiliary_data_dir="/a"):
        assert len(cfg) == 2
        assert cfg["output_dir"] == Path("/o")
        assert cfg["auxiliary_data_dir"] == Path("/a")
    assert not cfg


def test_context_mapping_invalid_option():
    cfg = Config()
    assert not cfg
    msg = r"`invalid_config_option` is not a valid config parameter"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with cfg.context({"invalid_config_option": 1}):
            pass


def test_context_kwargs_invalid_option():
    cfg = Config()
    assert not cfg
    msg = r"`invalid_config_option` is not a valid config parameter"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with cfg.context(invalid_config_option=1):
            pass
