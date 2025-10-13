import os
import warnings
from collections.abc import MutableMapping
from pathlib import Path
from textwrap import dedent

import pytest

import esmvalcore
import esmvalcore.config._config_object
from esmvalcore.config import CFG, Config, Session
from esmvalcore.config._config_object import DEFAULT_CONFIG_DIR
from esmvalcore.exceptions import (
    ESMValCoreDeprecationWarning,
    InvalidConfigParameter,
)
from tests.integration.test_main import arguments


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


def test_set_bad_item():
    config = Config({"output_dir": "config"})
    with pytest.raises(InvalidConfigParameter) as err_exc:
        config["bad_item"] = 47

    assert str(err_exc.value) == "`bad_item` is not a valid config parameter."


def test_config_init():
    config = Config()
    assert isinstance(config, MutableMapping)


# TODO: remove in v2.14.0
def test_load_from_file(monkeypatch):
    default_config_file = DEFAULT_CONFIG_DIR / "config-user.yml"
    config = Config()
    assert not config
    with pytest.warns(ESMValCoreDeprecationWarning):
        config.load_from_file(default_config_file)
    assert config


# TODO: remove in v2.14.0
def test_load_from_file_filenotfound(monkeypatch, tmp_path):
    """Test `Config.load_from_file`."""
    config = Config()
    assert not config

    expected_path = (
        tmp_path / "nonexistent_config_dir" / "not_existent_file.yml"
    )
    msg = f"Config file '{expected_path}' does not exist"
    with pytest.raises(FileNotFoundError, match=msg):
        config.load_from_file("not_existent_file.yml")


# TODO: remove in v2.14.0
def test_load_from_file_invalidconfigparameter(monkeypatch, tmp_path):
    """Test `Config.load_from_file`."""
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "test.yml"
    cfg_path.write_text("invalid_param: 42")

    config = Config()
    assert not config

    msg = (
        f"Failed to parse user configuration file {cfg_path}: `invalid_param` "
        f"is not a valid config parameter."
    )
    with pytest.raises(InvalidConfigParameter, match=msg):
        config.load_from_file(cfg_path)


def test_config_key_error():
    config = Config()
    with pytest.raises(KeyError):
        config["invalid_key"]


def test_reload(cfg_default, monkeypatch, tmp_path):
    """Test `Config.reload`."""
    # TODO: remove in v2.14.0
    monkeypatch.delenv("_ESMVALTOOL_USER_CONFIG_FILE_", raising=False)

    monkeypatch.setattr(
        esmvalcore.config._config_object,
        "USER_CONFIG_DIR",
        tmp_path / "this" / "is" / "an" / "empty" / "dir",
    )
    cfg = Config()

    cfg.reload()

    assert cfg == cfg_default


def test_reload_fail(monkeypatch, tmp_path):
    """Test `Config.reload`."""
    # TODO: remove in v2.14.0
    monkeypatch.delenv("_ESMVALTOOL_USER_CONFIG_FILE_", raising=False)

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


def test_session_key_error():
    session = Session({})
    with pytest.raises(KeyError):
        session["invalid_key"]


# TODO: remove in v2.14.0
def test_session_config_dir():
    session = Session({"config_file": "/path/to/config.yml"})
    with pytest.warns(ESMValCoreDeprecationWarning):
        config_dir = session.config_dir
    assert config_dir == Path("/path/to")


TEST_GET_CFG_PATH = [
    (
        None,
        None,
        None,
        "{tmp_path}/nonexistent_config_dir/config-user.yml",
        False,
    ),
    (
        None,
        None,
        ("any_other_module", "--config_file=cli.yml"),
        "{tmp_path}/nonexistent_config_dir/config-user.yml",
        False,
    ),
    (
        None,
        None,
        ("esmvaltool", "run", "--max_parallel_tasks=4"),
        "{tmp_path}/nonexistent_config_dir/config-user.yml",
        True,
    ),
    (
        None,
        None,
        ("esmvaltool", "--config_file"),
        "{tmp_path}/nonexistent_config_dir/config-user.yml",
        True,
    ),
    (
        None,
        None,
        ("esmvaltool", "run", "--config_file=/cli.yml"),
        "/cli.yml",
        True,
    ),
    (
        None,
        None,
        ("esmvaltool", "run", "--config_file=/cli.yml"),
        "/cli.yml",
        True,
    ),
    (
        None,
        None,
        ("esmvaltool", "run", "--config-file", "/cli.yml"),
        "/cli.yml",
        True,
    ),
    (
        None,
        None,
        ("esmvaltool", "run", "--config-file=/cli.yml"),
        "/cli.yml",
        True,
    ),
    (
        None,
        None,
        ("esmvaltool", "run", "--config-file=relative_cli.yml"),
        "{tmp_path}/nonexistent_config_dir/relative_cli.yml",
        True,
    ),
    (
        None,
        None,
        ("esmvaltool", "run", "--config-file=existing_cfg.yml"),
        "existing_cfg.yml",
        True,
    ),
    (
        None,
        {"_ESMVALTOOL_USER_CONFIG_FILE_": "/env.yml"},
        ("esmvaltool", "run", "--config-file=/cli.yml"),
        "/env.yml",
        True,
    ),
    (
        None,
        {"_ESMVALTOOL_USER_CONFIG_FILE_": "/env.yml"},
        None,
        "/env.yml",
        True,
    ),
    (
        None,
        {"_ESMVALTOOL_USER_CONFIG_FILE_": "existing_cfg.yml"},
        ("esmvaltool", "run", "--config-file=/cli.yml"),
        "existing_cfg.yml",
        True,
    ),
    (
        "/filename.yml",
        {"_ESMVALTOOL_USER_CONFIG_FILE_": "/env.yml"},
        ("esmvaltool", "run", "--config-file=/cli.yml"),
        "/filename.yml",
        True,
    ),
    (
        "/filename.yml",
        None,
        ("esmvaltool", "run", "--config-file=/cli.yml"),
        "/filename.yml",
        True,
    ),
    ("/filename.yml", None, None, "/filename.yml", False),
    (
        "filename.yml",
        None,
        None,
        "{tmp_path}/nonexistent_config_dir/filename.yml",
        False,
    ),
    (
        "existing_cfg.yml",
        {"_ESMVALTOOL_USER_CONFIG_FILE_": "/env.yml"},
        ("esmvaltool", "run", "--config-file=/cli.yml"),
        "existing_cfg.yml",
        True,
    ),
]


# TODO: remove in v2.14.0
@pytest.mark.parametrize(
    ("filename", "env", "cli_args", "output", "env_var_set"),
    TEST_GET_CFG_PATH,
)
def test_get_config_user_path(
    filename,
    env,
    cli_args,
    output,
    env_var_set,
    monkeypatch,
    tmp_path,
):
    """Test `Config._get_config_user_path`."""
    output = output.format(tmp_path=tmp_path)
    monkeypatch.delenv("_ESMVALTOOL_USER_CONFIG_FILE_", raising=False)

    # Create empty test file
    monkeypatch.chdir(tmp_path)
    (tmp_path / "existing_cfg.yml").write_text("")

    if output == "existing_cfg.yml":
        output = tmp_path / "existing_cfg.yml"
    else:
        output = Path(output).expanduser()

    if env is not None:
        for key, val in env.items():
            monkeypatch.setenv(key, val)
    if cli_args is None:
        cli_args = ["python"]

    with arguments(*cli_args):
        config_path = Config._get_config_user_path(filename)
        if env_var_set:
            assert os.environ["_ESMVALTOOL_USER_CONFIG_FILE_"] == str(output)
        else:
            assert "_ESMVALTOOL_USER_CONFIG_FILE_" not in os.environ
    assert isinstance(config_path, Path)
    assert config_path == output


# TODO: remove in v2.14.0
def test_load_user_config_filenotfound(tmp_path):
    """Test `Config._load_user_config`."""
    expected_path = (
        tmp_path / "nonexistent_config_dir" / "not_existent_file.yml"
    )
    msg = f"Config file '{expected_path}' does not exist"
    with pytest.raises(FileNotFoundError, match=msg):
        Config._load_user_config("not_existent_file.yml")


# TODO: remove in v2.14.0
def test_load_user_config_no_exception():
    """Test `Config._load_user_config`."""
    Config._load_user_config("not_existent_file.yml", raise_exception=False)


# TODO: remove in v2.14.0
def test_load_user_config_invalidconfigparameter(monkeypatch, tmp_path):
    """Test `Config._load_user_config`."""
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "test.yml"
    cfg_path.write_text("invalid_param: 42")

    msg = (
        f"Failed to parse user configuration file {cfg_path}: `invalid_param` "
        f"is not a valid config parameter."
    )
    with pytest.raises(InvalidConfigParameter, match=msg):
        Config._load_user_config(cfg_path)


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


# TODO: remove in v2.14.0
def test_get_global_config_force_new_config(mocker, tmp_path, monkeypatch):
    """Test ``_get_global_config``."""
    monkeypatch.setenv("ESMVALTOOL_CONFIG_DIR", "/path/to/config/file")

    # Create invalid old config file to ensure that this is not used
    config_file = tmp_path / "old_config_user.yml"
    config_file.write_text("invalid_option: /new/output/dir")
    mocker.patch.object(
        esmvalcore.config._config_object.Config,
        "_get_config_user_path",
        return_value=config_file,
    )

    # No deprecation message should be raised
    # Note: _get_global_config will ignore the old config since
    # ESMVALTOOL_CONFIG_DIR is set, but not actually use its value since
    # esmvalcore.config._config_object.USER_CONFIG_DIR has already been set to
    # its default value when loading this module
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        esmvalcore.config._config_object._get_global_config()


# TODO: remove in v2.14.0
def test_get_global_config_deprecated(mocker, tmp_path, monkeypatch):
    """Test ``_get_global_config``."""
    monkeypatch.delenv("ESMVALTOOL_CONFIG_DIR", raising=False)

    config_file = tmp_path / "old_config_user.yml"
    config_file.write_text("output_dir: /new/output/dir")
    mocker.patch.object(
        esmvalcore.config._config_object.Config,
        "_get_config_user_path",
        return_value=config_file,
    )
    with pytest.warns(ESMValCoreDeprecationWarning):
        cfg = esmvalcore.config._config_object._get_global_config()

    assert cfg["output_dir"] == Path("/new/output/dir")


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
    cfg["search_esgf"] = "when_missing"

    cfg.load_from_dirs(config_dirs)

    assert cfg["output_file_type"] == output_file_type
    assert cfg["rootpath"] == rootpath
    assert cfg["search_esgf"] == "never"


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
    cfg["search_esgf"] = "when_missing"

    cfg.update_from_dirs(config_dirs)

    if output_file_type is None:
        assert "output_file_type" not in cfg
    else:
        assert cfg["output_file_type"] == output_file_type
    assert cfg["rootpath"] == rootpath
    assert cfg["search_esgf"] == "when_missing"


def test_nested_update():
    """Test `Config.update_from_dirs`."""
    cfg = Config()
    assert not cfg

    cfg["drs"] = {"X": "x", "Z": "z"}
    cfg["search_esgf"] = "when_missing"

    cfg.nested_update({"drs": {"Y": "y", "X": "xx"}, "max_years": 1})

    assert len(cfg) == 3
    assert cfg["drs"] == {"Y": "y", "X": "xx", "Z": "z"}
    assert cfg["search_esgf"] == "when_missing"
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
