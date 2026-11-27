"""Progress bars for use with Dask."""

from __future__ import annotations

import contextvars
import datetime as dt
import threading
import time
from typing import TYPE_CHECKING, Any

import dask
import dask.diagnostics
import dask.utils
import distributed
import distributed.diagnostics.progressbar
import rich.progress
import rich.text
from loguru import logger

from esmvalcore.config._logging import _console_prefix

if TYPE_CHECKING:
    import contextlib
    from collections.abc import Iterable

    from dask.delayed import Delayed

    from esmvalcore.config import Session


class _PrefixColumn(rich.progress.ProgressColumn):
    """Show the same start of the line as the log messages."""

    def __init__(self) -> None:
        super().__init__()
        time_and_level, task = _console_prefix()
        self._text = rich.text.Text(time_and_level)
        if task:
            self._text.append(" ")
            self._text.append(f"{task}:", style="cyan")

    def render(self, task: rich.progress.Task) -> rich.text.Text:  # noqa: ARG002
        return self._text


def _rich_progress(**kwargs: Any) -> rich.progress.Progress:
    """Create a progress bar that is aligned with the log messages."""
    return rich.progress.Progress(
        _PrefixColumn(),
        rich.progress.TaskProgressColumn(),
        rich.progress.BarColumn(bar_width=80),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TimeElapsedColumn(),
        redirect_stdout=False,
        redirect_stderr=False,
        **kwargs,
    )


class RichProgressBar(dask.diagnostics.Callback):
    """Progress bar using `rich` for the Dask threaded scheduler."""

    # Adapted from https://github.com/dask/dask/blob/0f3e5ff6e642e7661b3f855bfd192a6f6fb83b49/dask/diagnostics/progress.py#L32-L153
    def __init__(self):
        self.progress = _rich_progress()
        self.task = self.progress.add_task(description="progress")
        self._dt = 0.1
        self._state = None
        self._running = False
        self._timer = None

    def _start(self, dsk):  # noqa: ARG002
        self._state = None
        # Start background thread
        self._running = True
        self._timer = threading.Thread(target=self._timer_func)
        self._timer.daemon = True
        self._timer.start()

    def _start_state(self, dsk, state):  # noqa: ARG002
        self.progress.start()
        total = sum(
            len(state[k]) for k in ["ready", "waiting", "running", "finished"]
        )
        self.progress.update(self.task, total=total)

    def _pretask(self, key, dsk, state):  # noqa: ARG002
        self._state = state

    def _finish(self, dsk, state, errored):  # noqa: ARG002
        self._running = False
        self._timer.join()
        self._draw_bar()
        self.progress.stop()

    def _timer_func(self):
        """Background thread for updating the progress bar."""
        while self._running:
            self._draw_bar()
            time.sleep(self._dt)

    def _draw_bar(self):
        state = self._state
        completed = len(state["finished"]) if state else 0
        self.progress.update(self.task, completed=completed)


class RichDistributedProgressBar(
    distributed.diagnostics.progressbar.TextProgressBar,
):
    """Progress bar using `rich` for the Dask distributed scheduler."""

    def __init__(self, keys) -> None:  # noqa: ANN001
        self.progress = _rich_progress()
        self.progress.start()
        self.task_id = self.progress.add_task(description="progress")
        super().__init__(keys)

    def _draw_bar(
        self,
        remaining: int,
        all: int,  # noqa: A002 # pylint: disable=redefined-builtin
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        completed = all - remaining
        self.progress.update(self.task_id, completed=completed, total=all)

    def _draw_stop(self, **kwargs):
        if kwargs.get("status") == "finished":
            self._draw_bar(remaining=0, all=self.progress.tasks[0].total)
        self.progress.stop()


class ProgressLogger(dask.diagnostics.ProgressBar):
    """Progress logger for the Dask threaded scheduler."""

    def __init__(
        self,
        log_interval: str | float = "1s",
    ) -> None:
        # The bar is drawn from another thread, so keep a copy of the current
        # context to show the task that logged the messages.
        self._context = contextvars.copy_context()
        self._log_interval = dask.utils.parse_timedelta(
            log_interval,
            default="s",
        )
        self._prev_elapsed = 0.0
        interval = dask.utils.parse_timedelta("1s", default="s")
        super().__init__(dt=interval)
        self._file = None

    def _draw_bar(self, frac: float, elapsed: float) -> None:
        if (elapsed - self._prev_elapsed) < self._log_interval and frac != 1.0:
            return
        self._prev_elapsed = elapsed
        pbar = "#" * int(self._width * frac)
        percent = int(100 * frac)
        elapsed_fmt = dask.utils.format_time(elapsed)
        msg = (
            f"[{pbar:<{self._width}}] | {percent:3}% Completed | {elapsed_fmt}"
        )
        self._context.run(logger.info, msg)


class DistributedProgressLogger(
    distributed.diagnostics.progressbar.TextProgressBar,
):
    """Progress logger for the Dask distributed scheduler."""

    def __init__(
        self,
        keys,  # noqa: ANN001
        log_interval: str | float = "1s",
    ) -> None:
        # The bar is drawn from another thread, so keep a copy of the current
        # context to show the task that logged the messages.
        self._context = contextvars.copy_context()
        self._log_interval = dask.utils.parse_timedelta(
            log_interval,
            default="s",
        )
        self._prev_elapsed = 0.0
        super().__init__(keys, interval="1s")

    def _draw_bar(
        self,
        remaining: int,
        all: int,  # noqa: A002 # pylint: disable=redefined-builtin
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        frac = (1 - remaining / all) if all else 1.0
        if (
            self.elapsed - self._prev_elapsed
        ) < self._log_interval and frac != 1.0:
            return
        self._prev_elapsed = self.elapsed
        pbar = "#" * int(self.width * frac)
        percent = int(100 * frac)
        elapsed = dask.utils.format_time(self.elapsed)
        msg = f"[{pbar:<{self.width}}] | {percent:3}% Completed | {elapsed}"
        self._context.run(logger.info, msg)

    def _draw_stop(self, **kwargs):
        pass


def _compute_with_progress(
    delayeds: Iterable[Delayed],
    session: Session,
) -> None:
    """Compute delayeds while displaying a progress bar."""
    use_distributed = True
    try:
        distributed.get_client()
    except ValueError:
        use_distributed = False

    log_progress_interval = session["logging"]["log_progress_interval"]
    if isinstance(log_progress_interval, (str, dt.timedelta)):
        log_progress_interval = dask.utils.parse_timedelta(
            log_progress_interval,
        )
    if session["max_parallel_tasks"] != 1 and log_progress_interval == 0.0:
        # Enable progress logging if `max_parallel_tasks` > 1 to avoid clutter.
        log_progress_interval = 10.0

    # There are three possible options, depending on the value of
    # CFG["log_progress_interval"]:
    # < 0: no progress reporting
    # = 0: show progress bar
    # > 0: log progress at this interval
    if log_progress_interval < 0.0:
        dask.compute(delayeds)
    elif use_distributed:
        futures = dask.persist(delayeds)
        futures = distributed.client.futures_of(futures)
        if log_progress_interval == 0.0:
            RichDistributedProgressBar(futures)
        else:
            DistributedProgressLogger(
                futures,
                log_interval=log_progress_interval,
            )
        dask.compute(futures)
    else:
        if log_progress_interval == 0.0:
            ctx: contextlib.AbstractContextManager = RichProgressBar()
        else:
            ctx = ProgressLogger(
                log_interval=log_progress_interval,
            )
        with ctx:
            dask.compute(delayeds)
