"""ESMValtool task definition."""

import abc
import contextlib
import datetime
import importlib
import logging
import multiprocessing
import numbers
import os
import pprint
import subprocess
import sys
import textwrap
import threading
import time
from copy import deepcopy
from pathlib import Path, PosixPath
from shutil import which

import dask
import psutil
import yaml
from distributed import Client

from ._citation import _write_citation_files
from ._provenance import TrackedFile, get_task_provenance
from .config._dask import get_distributed_client
from .config._diagnostics import DIAGNOSTICS, TAGS


def path_representer(dumper, data):
    """For printing pathlib.Path objects in yaml files."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))


yaml.representer.SafeRepresenter.add_representer(Path, path_representer)
yaml.representer.SafeRepresenter.add_representer(PosixPath, path_representer)

logger = logging.getLogger(__name__)

DATASET_KEYS = {
    "mip",
}


def _get_resource_usage(process, start_time, children=True):
    """Get resource usage."""
    # yield header first
    entries = [
        "Date and time (UTC)",
        "Real time (s)",
        "CPU time (s)",
        "CPU (%)",
        "Memory (GB)",
        "Memory (%)",
        "Disk read (GB)",
        "Disk write (GB)",
    ]
    fmt = "{}\t" * len(entries[:-1]) + "{}\n"
    yield (fmt.format(*entries), 0.0)

    # Compute resource usage
    gigabyte = float(2**30)
    precision = [1, 1, None, 1, None, 3, 3]
    cache = {}
    max_memory = 0.0
    try:
        process.io_counters()
    except AttributeError:
        counters_available = False
    else:
        counters_available = True
    while process.is_running():
        try:
            if children:
                # Include child processes
                processes = process.children(recursive=True)
                processes.append(process)
            else:
                processes = [process]

            # Update resource usage
            for proc, cached_value in cache.items():
                # Set cpu percent and memory usage to 0 for old processes
                if proc not in processes:
                    cached_value[1] = 0
                    cached_value[2] = 0
                    cached_value[3] = 0
            for proc in processes:
                # Update current processes
                cache[proc] = [
                    proc.cpu_times().user + proc.cpu_times().system,
                    proc.cpu_percent(),
                    proc.memory_info().rss / gigabyte,
                    proc.memory_percent(),
                    (
                        proc.io_counters().read_bytes / gigabyte
                        if counters_available
                        else float("nan")
                    ),
                    (
                        proc.io_counters().write_bytes / gigabyte
                        if counters_available
                        else float("nan")
                    ),
                ]
        except (OSError, psutil.AccessDenied, psutil.NoSuchProcess):
            # Try again if an error occurs because some process died
            continue

        # Create and yield log entry
        entries = [sum(entry) for entry in zip(*cache.values(), strict=False)]
        entries.insert(0, time.time() - start_time)
        entries = [
            round(entry, p)
            for entry, p in zip(entries, precision, strict=False)
        ]
        entries.insert(0, datetime.datetime.now(datetime.UTC))
        max_memory = max(max_memory, entries[4])
        yield (fmt.format(*entries), max_memory)


@contextlib.contextmanager
def resource_usage_logger(pid, filename, interval=1, children=True):
    """Log resource usage."""
    halt = threading.Event()

    def _log_resource_usage():
        """Write resource usage to file."""
        process = psutil.Process(pid)
        start_time = time.time()
        with open(filename, "w", encoding="utf-8") as file:
            for msg, max_mem in _get_resource_usage(
                process,
                start_time,
                children,
            ):
                file.write(msg)
                time.sleep(interval)
                if halt.is_set():
                    logger.info(
                        "Maximum memory used (estimate): %.1f GB",
                        max_mem,
                    )
                    logger.info(
                        "Sampled every second. It may be inaccurate if short "
                        "but high spikes in memory consumption occur.",
                    )
                    return

    thread = threading.Thread(target=_log_resource_usage)
    thread.start()
    try:
        yield
    finally:
        halt.set()
        thread.join()


def _py2ncl(value, var_name=""):
    """Format a structure of Python list/dict/etc items as NCL."""
    txt = var_name + " = " if var_name else ""
    if value is None:
        txt += "_Missing"
    elif isinstance(value, (str, Path)):
        txt += f'"{value}"'
    elif isinstance(value, (list, tuple)):
        if not value:
            txt += "_Missing"
        else:
            if isinstance(value[0], numbers.Real):
                type_ = numbers.Real
            else:
                type_ = type(value[0])
            if any(not isinstance(v, type_) for v in value):
                msg = f"NCL array cannot be mixed type: {value}"
                raise ValueError(
                    msg,
                )
            txt += "(/{}/)".format(", ".join(_py2ncl(v) for v in value))
    elif isinstance(value, dict):
        if not var_name:
            msg = f"NCL does not support nested dicts: {value}"
            raise ValueError(
                msg,
            )
        txt += "True\n"
        for key in value:
            txt += f"{var_name}@{key} = {_py2ncl(value[key])}\n"
    else:
        txt += str(value)
    return txt


def write_ncl_settings(settings, filename, mode="wt"):
    """Write a dictionary with generic settings to NCL file."""
    logger.debug("Writing NCL configuration file %s", filename)

    def _ncl_type(value):
        """Convert some Python types to NCL types."""
        typemap = {
            bool: "logical",
            str: "string",
            float: "double",
            int: "int64",
            dict: "logical",
        }
        for type_, ncl_type in typemap.items():
            if isinstance(value, type_):
                return ncl_type
        msg = f"Unable to map {type(value)} to an NCL type"
        raise ValueError(msg)

    lines = []

    # ignore some settings for NCL diagnostic
    ignore_settings = [
        "profile_diagnostic",
    ]
    for sett in ignore_settings:
        settings_copy = dict(settings)
        if "diag_script_info" not in settings_copy:
            settings.pop(sett, None)
        else:
            settings_copy["diag_script_info"].pop(sett, None)

    for var_name, value in sorted(settings_copy.items()):
        if isinstance(value, (list, tuple)):
            # Create an NCL list that can span multiple files
            lines.append(
                f'if (.not. isdefined("{var_name}")) then\n'
                f'  {var_name} = NewList("fifo")\n'
                "end if\n",
            )
            for item in value:
                lines.append(
                    f"ListAppend({var_name}, new(1, {_ncl_type(item)}))\n"
                    f"i = ListCount({var_name}) - 1",
                )
                lines.append(_py2ncl(item, var_name + "[i]"))
        else:
            # Create an NCL variable that overwrites previous variables
            lines.append(
                f'if (isvar("{var_name}")) then\n'
                f"  delete({var_name})\n"
                "end if\n",
            )
            lines.append(_py2ncl(value, var_name))

    with open(filename, mode, encoding="utf-8") as file:
        file.write("\n".join(lines))
        file.write("\n")


class BaseTask:
    """Base class for defining task classes."""

    def __init__(self, ancestors=None, name="", products=None):
        """Initialize task."""
        self.ancestors = [] if ancestors is None else ancestors
        self.products = set() if products is None else set(products)
        self.output_files = None
        self.name = name
        self.activity = None
        self.priority = 0
        self.scheduler_lock = None

    def initialize_provenance(self, recipe_entity):
        """Initialize task provenance activity."""
        if self.activity is not None:
            msg = f"Provenance of {self} already initialized"
            raise ValueError(
                msg,
            )
        self.activity = get_task_provenance(self, recipe_entity)

    def flatten(self):
        """Return a flattened set of all ancestor tasks and task itself."""
        tasks = TaskSet()
        for task in self.ancestors:
            tasks.update(task.flatten())
        tasks.add(self)
        return tasks

    def run(self, input_files=None):
        """Run task."""
        if not self.output_files:
            if input_files is None:
                input_files = []
            for task in self.ancestors:
                input_files.extend(task.run())
            logger.info(
                "Starting task %s in process [%s]",
                self.name,
                os.getpid(),
            )
            start = datetime.datetime.now()
            self.output_files = self._run(input_files)
            runtime = datetime.datetime.now() - start
            logger.info(
                "Successfully completed task %s (priority %s) in %s",
                self.name,
                self.priority,
                runtime,
            )

        return self.output_files

    @abc.abstractmethod
    def _run(self, input_files):
        """Run task."""

    def get_product_attributes(self) -> dict:
        """Return a mapping of product attributes."""
        return {
            product.filename: product.attributes
            for product in sorted(self.products)
        }

    def print_ancestors(self):
        """Return a nicely formatted description."""
        return "ancestors:\n{}".format(
            "\n\n".join(
                textwrap.indent(str(task), prefix="  ")
                for task in self.ancestors
            )
            if self.ancestors
            else "None",
        )

    def __repr__(self):
        """Return canonical string representation."""
        return f"{self.__class__.__name__}({self.name!r})"


class ResumeTask(BaseTask):
    """Task for reusing preprocessor output files from a previous run."""

    def __init__(self, prev_preproc_dir, preproc_dir, name):
        """Create a resume task."""
        # Set the path to the file resulting from running this task
        self._metadata_file = preproc_dir / "metadata.yml"

        # Reconstruct output
        prev_metadata_file = prev_preproc_dir / "metadata.yml"
        with prev_metadata_file.open("r", encoding="utf-8") as file:
            prev_metadata = yaml.safe_load(file)

        products = set()
        for prov_filename, attributes in prev_metadata.items():
            # Update the filename in case the output directory was moved
            # since the original run
            filename = str(prev_preproc_dir / Path(prov_filename).name)
            attributes["filename"] = filename
            product = TrackedFile(
                filename,
                attributes,
                prov_filename=prov_filename,
            )
            products.add(product)

        super().__init__(ancestors=None, name=name, products=products)

    def _run(self, _):
        """Return the result of a previous run."""
        metadata = self.get_product_attributes()

        # Write metadata to file
        self._metadata_file.parent.mkdir(parents=True)
        with self._metadata_file.open("w", encoding="utf-8") as file:
            yaml.safe_dump(metadata, file)

        return [str(self._metadata_file)]


class DiagnosticError(Exception):
    """Error in diagnostic."""


class DiagnosticTask(BaseTask):
    """Task for running a diagnostic."""

    def __init__(self, script, settings, output_dir, ancestors=None, name=""):
        """Create a diagnostic task."""
        super().__init__(ancestors=ancestors, name=name)
        self.script = script
        self.settings = settings
        self.output_dir = output_dir
        self.cmd = self._initialize_cmd()
        self.env = self._initialize_env()
        self.log = Path(settings["run_dir"]) / "log.txt"
        self.resource_log = Path(settings["run_dir"]) / "resource_usage.txt"

    def _initialize_cmd(self):
        """Create an executable command from script."""
        diagnostics_root = DIAGNOSTICS.scripts
        script = self.script

        # Check if local diagnostic path exists
        script_file = Path(script).expanduser().absolute()
        err_msg = f"Cannot execute script '{script}' ({script_file})"
        if not script_file.is_file():
            logger.debug(
                "No local diagnostic script found. Attempting to load the script from the base repository.",
            )
            # Check if esmvaltool package is available
            if importlib.util.find_spec("esmvaltool") is None:
                logger.warning(
                    "The 'esmvaltool' package cannot be found. Please ensure it is installed.",
                )

            # Try diagnostics_root
            script_file = (
                diagnostics_root / Path(script).expanduser()
            ).absolute()
        if not script_file.is_file():
            msg = f"{err_msg}: file does not exist."
            raise DiagnosticError(msg)

        cmd = []

        interpreters = {
            "jl": "julia",
            "ncl": "ncl",
            "py": "python",
            "r": "Rscript",
        }
        args = {
            "ncl": ["-n", "-p"],
        }
        if self.settings["profile_diagnostic"]:
            profile_file = Path(self.settings["run_dir"], "profile.json")
            args["py"] = ["-m", "vprof", "-o", str(profile_file), "-c", "c"]

        ext = script_file.suffix.lower()[1:]
        if ext in interpreters:
            if ext == "py" and sys.executable:
                interpreter = sys.executable
            else:
                interpreter = which(interpreters[ext])
            if interpreter is None:
                msg = (
                    f"{err_msg}: program '{interpreters[ext]}' not installed."
                )
                raise DiagnosticError(
                    msg,
                )
            cmd.append(interpreter)
        elif not os.access(script_file, os.X_OK):
            msg = (
                f"{err_msg}: non-executable file with unknown extension "
                f"'{script_file.suffix}'."
            )
            raise DiagnosticError(
                msg,
            )

        cmd.extend(args.get(ext, []))
        cmd.append(str(script_file))

        return cmd

    def _initialize_env(self):
        """Create an environment for executing script."""
        ext = Path(self.script).suffix.lower()
        env = {}
        if ext in (".py", ".jl"):
            # Set non-interactive matplotlib backend
            env["MPLBACKEND"] = "Agg"
        if ext in (".r", ".ncl"):
            # Make diag_scripts path available to diagostic script
            env["diag_scripts"] = str(DIAGNOSTICS.scripts)
        if ext == ".jl":
            # Set the julia virtual environment
            env["JULIA_LOAD_PATH"] = "{}:{}".format(
                DIAGNOSTICS.path / "install" / "Julia",
                os.environ.get("JULIA_LOAD_PATH", ""),
            )
        return env

    def write_settings(self):
        """Write settings to file."""
        run_dir = Path(self.settings["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)

        # ignore some settings for diagnostic
        ignore_settings = [
            "profile_diagnostic",
        ]
        for sett in ignore_settings:
            settings_copy = dict(self.settings)
            settings_copy.pop(sett, None)

        filename = run_dir / "settings.yml"
        filename.write_text(yaml.safe_dump(settings_copy, sort_keys=False))

        # If running an NCL script:
        if Path(self.script).suffix.lower() == ".ncl":
            # Also write an NCL file and return the name of that instead.
            return self._write_ncl_settings()

        return str(filename)

    def _write_ncl_settings(self):
        """Write settings to NCL file."""
        filename = Path(self.settings["run_dir"]) / "settings.ncl"

        config_user_keys = {
            "run_dir",
            "plot_dir",
            "work_dir",
            "output_file_type",
            "log_level",
        }
        settings = {"diag_script_info": {}, "config_user_info": {}}
        for key, value in self.settings.items():
            if key in config_user_keys:
                settings["config_user_info"][key] = value
            elif not isinstance(value, dict):
                settings["diag_script_info"][key] = value
            else:
                settings[key] = value

        write_ncl_settings(settings, filename)

        return filename

    def _control_ncl_execution(self, process, lines):
        """Check if an error has occurred in an NCL script.

        Apparently NCL does not automatically exit with a non-zero exit
        code if an error occurs, so we take care of that here.
        """
        ignore_warnings = [
            warning.strip()
            for warning in self.settings.get("ignore_ncl_warnings", [])
        ]

        errors = ["error:", "fatal:"]
        if self.settings["exit_on_ncl_warning"]:
            errors.append("warning:")

        msg = (
            f"An error occurred during execution of NCL script {self.script}, "
            f"see the log in {self.log}"
        )

        warned = False
        for line in lines:
            if line.strip() in ignore_warnings:
                continue
            if "warning:" in line:
                logger.warning("NCL: %s", line)
                warned = True
            for error in errors:
                if error in line:
                    logger.error(msg)
                    logger.error("NCL: %s", line)
                    try:
                        process.kill()
                    except OSError:  # ignore error if process already exited
                        pass
                    else:
                        logger.error("Killed process.")
                    raise DiagnosticError(msg)

        if warned:
            logger.warning(
                "There were warnings during the execution of NCL script %s, "
                "for details, see the log %s",
                self.script,
                self.log,
            )

    def _start_diagnostic_script(self, cmd, env):
        """Start the diagnostic script."""
        logger.info("Running command %s", cmd)
        logger.debug("in environment\n%s", pprint.pformat(env))
        cwd = self.settings["run_dir"]
        logger.debug("in current working directory: %s", cwd)
        logger.info("Writing output to %s", self.output_dir)
        logger.info("Writing plots to %s", self.settings["plot_dir"])
        logger.info("Writing log to %s", self.log)

        rerun_msg = f"cd {cwd}; "
        if env:
            rerun_msg += " ".join(f'{k}="{env[k]}"' for k in env)
        if "vprof" in cmd:
            script_args = ' "' + cmd[-1] + '"'
            rerun_msg += " " + " ".join(cmd[:-1]) + script_args
        else:
            rerun_msg += " " + " ".join(cmd)
        logger.info("To re-run this diagnostic script, run:\n%s", rerun_msg)

        complete_env = dict(os.environ)
        complete_env.update(env)

        return subprocess.Popen(
            cmd,
            bufsize=2**20,  # Use a large buffer to prevent NCL crash
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=complete_env,
        )

    def _run(self, input_files):
        """Run the diagnostic script."""
        if self.script is None:  # Run only preprocessor
            return []

        ext = Path(self.script).suffix.lower()
        if ext == ".ncl":
            self.settings["input_files"] = [
                f
                for f in input_files
                if f.endswith(".ncl") or os.path.isdir(f)
            ]
        else:
            self.settings["input_files"] = [
                f
                for f in input_files
                if f.endswith(".yml") or os.path.isdir(f)
            ]

        env = dict(self.env)
        cmd = list(self.cmd)
        settings_file = self.write_settings()
        if ext == ".ncl":
            env["settings"] = settings_file
        elif self.settings["profile_diagnostic"]:
            script_file = cmd.pop()
            combo_with_settings = script_file + " " + str(settings_file)
            cmd.append(combo_with_settings)
        else:
            cmd.append(settings_file)

        process = self._start_diagnostic_script(cmd, env)

        returncode = None

        with (
            resource_usage_logger(process.pid, self.resource_log),
            open(self.log, "ab") as log,
        ):
            last_line = [""]
            while returncode is None:
                returncode = process.poll()
                txt = process.stdout.read()
                log.write(txt)

                # Check if an error occurred in an NCL script
                # Last line is treated separately to avoid missing
                # error messages spread out over multiple lines.
                if ext == ".ncl":
                    txt = txt.decode(encoding="utf-8", errors="ignore")
                    lines = txt.split("\n")
                    self._control_ncl_execution(process, last_line + lines)
                    last_line = lines[-1:]

                # wait, but not long because the stdout buffer may fill up:
                # https://docs.python.org/3.6/library/subprocess.html#subprocess.Popen.stdout
                time.sleep(0.001)

        if returncode == 0:
            logger.debug("Script %s completed successfully", self.script)
            self._collect_provenance()
            return [self.output_dir]

        msg = (
            f"Diagnostic script {self.script} failed with return code {returncode}. See the log "
            f"in {self.log}"
        )
        raise DiagnosticError(
            msg,
        )

    def _collect_provenance(self):
        """Process provenance information provided by the diagnostic script."""
        provenance_file = (
            Path(self.settings["run_dir"]) / "diagnostic_provenance.yml"
        )
        if not provenance_file.is_file():
            logger.warning(
                "No provenance information was written to %s. Unable to "
                "record provenance for files created by diagnostic script %s "
                "in task %s",
                provenance_file,
                self.script,
                self.name,
            )
            return

        logger.debug("Collecting provenance from %s", provenance_file)
        start = time.time()
        table = yaml.safe_load(provenance_file.read_text(encoding="utf-8"))

        ignore = (
            "auxiliary_data_dir",
            "exit_on_ncl_warning",
            "input_files",
            "log_level",
            "output_file_type",
            "plot_dir",
            "profile_diagnostic",
            "recipe",
            "run_dir",
            "version",
            "write_ncl_interface",
            "work_dir",
        )
        attrs = {
            "script_file": self.script,
        }
        for key in self.settings:
            if key not in ignore:
                attrs[key] = self.settings[key]

        ancestor_products = {
            str(p.filename): p for a in self.ancestors for p in a.products
        }

        valid = True
        for filename, orig_attributes in table.items():
            # copy to avoid updating other entries if file contains anchors
            attributes = deepcopy(orig_attributes)
            ancestor_files = attributes.pop("ancestors", [])
            if not ancestor_files:
                logger.warning(
                    "No ancestor files specified for recording provenance of "
                    "%s, created by diagnostic script %s in task %s",
                    filename,
                    self.script,
                    self.name,
                )
                valid = False
            ancestors = set()
            if isinstance(ancestor_files, str):
                logger.warning(
                    "Ancestor file(s) %s specified for recording provenance "
                    "of %s, created by diagnostic script %s in task %s is "
                    "a string but should be a list of strings",
                    ancestor_files,
                    filename,
                    self.script,
                    self.name,
                )
                ancestor_files = [ancestor_files]
            for ancestor_file in ancestor_files:
                if ancestor_file in ancestor_products:
                    ancestors.add(ancestor_products[ancestor_file])
                else:
                    valid = False
                    logger.warning(
                        "Invalid ancestor file %s specified for recording "
                        "provenance of %s, created by diagnostic script %s "
                        "in task %s",
                        ancestor_file,
                        filename,
                        self.script,
                        self.name,
                    )

            attributes.update(deepcopy(attrs))

            TAGS.replace_tags_in_dict(attributes)

            product = TrackedFile(filename, attributes, ancestors)
            product.initialize_provenance(self.activity)
            _write_citation_files(product.filename, product.provenance)
            product.save_provenance()
            self.products.add(product)

        if not valid:
            logger.warning(
                "Valid ancestor files for diagnostic script %s in task %s "
                "are:\n%s",
                self.script,
                self.name,
                "\n".join(ancestor_products),
            )
        logger.debug(
            "Collecting provenance of task %s took %.1f seconds",
            self.name,
            time.time() - start,
        )

    def __repr__(self):
        """Get human readable description."""
        settings_string = pprint.pformat(self.settings)
        return (
            f"{self.__class__.__name__}: {self.name}\n"
            f"script: {self.script}\n"
            f"settings:\n{settings_string}\n"
            f"{self.print_ancestors()}\n"
        )


def available_cpu_count() -> int:
    """Return the number of available CPU cores."""
    if hasattr(os, "sched_getaffinity"):
        # Not available on OSX.
        return len(os.sched_getaffinity(0))
    if count := os.cpu_count():
        return count
    return 1


class TaskSet(set):
    """Container for tasks."""

    def flatten(self) -> "TaskSet":
        """Flatten the list of tasks."""
        return TaskSet(t for task in self for t in task.flatten())

    def get_independent(self) -> "TaskSet":
        """Return a set of independent tasks."""
        independent_tasks = TaskSet()
        all_tasks = self.flatten()
        for task in all_tasks:
            if not any(task in t.ancestors for t in all_tasks):
                independent_tasks.add(task)
        return independent_tasks

    def run(self, max_parallel_tasks: int | None = None) -> None:
        """Run tasks.

        Parameters
        ----------
        max_parallel_tasks : int
            Number of processes to run. If `1`, run the tasks sequentially.
        """
        with get_distributed_client() as client:
            if client is None:
                address = None
            else:
                address = client.scheduler.address
                for task in self.flatten():
                    if (
                        isinstance(task, DiagnosticTask)
                        and Path(task.script).suffix.lower() == ".py"
                    ):
                        # Only insert the scheduler address if running a
                        # Python script.
                        task.settings["scheduler_address"] = address

            if max_parallel_tasks == 1:
                self._run_sequential()
            else:
                self._run_parallel(address, max_parallel_tasks)

    def _run_sequential(self) -> None:
        """Run tasks sequentially."""
        n_tasks = len(self.flatten())
        logger.info("Running %s tasks sequentially", n_tasks)

        tasks = self.get_independent()
        for task in sorted(tasks, key=lambda t: t.priority):
            task.run()

    def _get_dask_config(self, max_parallel_tasks: int) -> dict:
        """Configure the threaded Dask scheduler.

        Configure the threaded Dask scheduler to use a reasonable number
        of threads when the user has not done so. We will run multiple
        processes, each of which will start its own scheduler with
        `num_workers` threads. To avoid too much parallelism, we would like to
        create n_threads = n_cpu_cores / n_processes.
        """
        # pylint: disable=import-outside-toplevel
        from esmvalcore.preprocessor import PreprocessingTask  # noqa: PLC0415

        if dask.config.get("scheduler", "threads") not in (
            "threads",
            "threading",
        ):
            # No need to do anything when not using the threaded scheduler
            # https://github.com/dask/dask/blob/3504bcc89f7a937b2d48306a17b8eeff57b1e5ae/dask/base.py#L1027-L1050
            return {}
        if dask.config.get("num_workers", None) is not None:
            # No need to do anything when the user has configured "num_workers".
            return {}

        n_preproc_tasks = sum(
            isinstance(t, PreprocessingTask) for t in self.flatten()
        )
        if n_preproc_tasks == 0:
            # No need to do anything when we are not running PreprocessingTasks.
            return {}

        n_available_cpu_cores = available_cpu_count()
        n_threaded_dask_schedulers = min(n_preproc_tasks, max_parallel_tasks)
        n_workers = max(
            1,
            round(n_available_cpu_cores / n_threaded_dask_schedulers),
        )
        logger.info(
            "Using the threaded Dask scheduler with %s worker threads per "
            "preprocessing task. "
            "See https://docs.esmvaltool.org/projects/ESMValCore/en/"
            "latest/quickstart/configure.html#f5 for more information.",
            n_workers,
        )
        return {"num_workers": n_workers}

    def _run_parallel(self, scheduler_address, max_parallel_tasks):
        """Run tasks in parallel."""
        scheduled = self.flatten()
        running = {}

        n_tasks = n_scheduled = len(scheduled)
        n_running = 0

        if max_parallel_tasks is None:
            max_parallel_tasks = available_cpu_count()
        max_parallel_tasks = min(max_parallel_tasks, n_tasks)
        logger.info(
            "Running %s tasks using %s processes",
            n_tasks,
            max_parallel_tasks,
        )

        dask_config = self._get_dask_config(max_parallel_tasks)

        def done(task):
            """Assume a task is done if it not scheduled or running."""
            return not (task in scheduled or task in running)

        with multiprocessing.Manager() as manager:
            # Use a lock to avoid overloading the Dask workers by making only
            # one :class:`esmvalcore.preprocessor.PreprocessingTask` submit its
            # data save task graph to the distributed scheduler at a time.
            #
            # See https://github.com/ESMValGroup/ESMValCore/issues/2609 for
            # additional detail.
            scheduler_lock = (
                None if scheduler_address is None else manager.Lock()
            )

            with (
                dask.config.set(dask_config),
                multiprocessing.Pool(processes=max_parallel_tasks) as pool,
            ):
                while scheduled or running:
                    # Submit new tasks to pool
                    for task in sorted(scheduled, key=lambda t: t.priority):
                        if len(running) >= max_parallel_tasks:
                            break
                        if all(done(t) for t in task.ancestors):
                            future = pool.apply_async(
                                _run_task,
                                [task, scheduler_address, scheduler_lock],
                            )
                            running[task] = future
                            scheduled.remove(task)

                    # Handle completed tasks
                    ready = {t for t in running if running[t].ready()}
                    for task in ready:
                        _copy_results(task, running[task])
                        running.pop(task)

                    # Wait if there are still tasks running
                    if running:
                        time.sleep(0.1)

                    # Log progress message
                    if (
                        len(scheduled) != n_scheduled
                        or len(running) != n_running
                    ):
                        n_scheduled, n_running = len(scheduled), len(running)
                        n_done = n_tasks - n_scheduled - n_running
                        logger.info(
                            "Progress: %s tasks running, %s tasks waiting for "
                            "ancestors, %s/%s done",
                            n_running,
                            n_scheduled,
                            n_done,
                            n_tasks,
                        )

                logger.info("Successfully completed all tasks.")
                pool.close()
                pool.join()


def _copy_results(task, future):
    """Update task with the results from the remote process."""
    task.output_files, task.products = future.get()


def _run_task(task, scheduler_address, scheduler_lock):
    """Run task and return the result."""
    if scheduler_address is None:
        client = contextlib.nullcontext()
    else:
        client = Client(scheduler_address)

    with client:
        task.scheduler_lock = scheduler_lock
        output_files = task.run()

    return output_files, task.products
