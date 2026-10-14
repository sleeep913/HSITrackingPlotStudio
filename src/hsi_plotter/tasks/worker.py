"""Reusable, cancellable background task infrastructure.

UI callbacks only submit handlers here; never embed scanning or computation
logic in Qt slots (NFR / 实施计划 原则 4).
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

TaskHandler = Callable[["ProgressReporter", "CancelToken"], object]


class TaskCancelled(RuntimeError):
    """Raised inside a handler when the user requested cancellation."""


class CancelToken:
    """Thread-safe cancellation flag shared with a running handler."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise TaskCancelled("任务已被取消")

    def check(self) -> bool:
        """``True`` when no cancellation has been requested."""
        return not self._event.is_set()


class ProgressReporter:
    """Reports progress back to the UI thread and raises on cancellation."""

    def __init__(self, emit: Callable[[int, str], None], token: CancelToken) -> None:
        self._emit = emit
        self._token = token

    def update(self, message: str = "", *, value: int | None = None) -> None:
        self._emit(value, message)
        self._token.raise_if_cancelled()

    def report(self, completed: int, total: int, message: str = "") -> None:
        total = max(int(total), 1)
        completed = min(max(int(completed), 0), total)
        self._emit(int(round(completed * 100 / total)), message)
        self._token.raise_if_cancelled()

    def notify(self, value: int | None = None, message: str = "") -> None:
        """Report progress without aborting the task.

        Handlers that keep already computed results on cancellation use this
        instead of :meth:`report` (FR-EVAL-006).
        """
        self._emit(value, message)


class _Worker(QRunnable):
    def __init__(self, task_id: str, name: str, handler: TaskHandler) -> None:
        super().__init__()
        self.task_id = task_id
        self.name = name
        self.token = CancelToken()
        self.signals = WorkerSignals()
        self._handler = handler

    def run(self) -> None:
        reporter = ProgressReporter(
            lambda value, message: self.signals.progress.emit(self.task_id, value, message),
            self.token,
        )
        try:
            self.signals.started.emit(self.task_id)
            result = self._handler(reporter, self.token)
        except TaskCancelled:
            self.signals.cancelled.emit(self.task_id)
        except Exception as exc:  # noqa: BLE001 - reported to the UI, never fatal
            self.signals.failed.emit(self.task_id, str(exc))
        else:
            self.signals.succeeded.emit(self.task_id, result)
        finally:
            self.signals.finished.emit(self.task_id)


class WorkerSignals(QObject):
    started = Signal(str)
    progress = Signal(str, int, str)
    succeeded = Signal(str, object)
    failed = Signal(str, str)
    cancelled = Signal(str)
    finished = Signal(str)


class TaskRunner(QObject):
    """Submits cancellable work to a thread pool and mirrors state as signals."""

    taskStarted = Signal(str, str)
    taskProgress = Signal(str, int, str)
    taskSucceeded = Signal(str, object)
    taskFailed = Signal(str, str)
    taskCancelled = Signal(str)
    taskFinished = Signal(str)
    busyChanged = Signal(bool)

    def __init__(self, parent: QObject | None = None, *, max_concurrent: int = 1) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(1, int(max_concurrent)))
        self._workers: dict[str, _Worker] = {}
        self._names: dict[str, str] = {}
        self._lock = threading.Lock()

    def submit(self, name: str, handler: TaskHandler) -> str:
        task_id = uuid.uuid4().hex
        worker = _Worker(task_id, name, handler)
        worker.signals.started.connect(self._on_started)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.succeeded.connect(self._on_succeeded)
        worker.signals.failed.connect(self._on_failed)
        worker.signals.cancelled.connect(self._on_cancelled)
        worker.signals.finished.connect(self._on_finished)
        with self._lock:
            self._workers[task_id] = worker
            self._names[task_id] = name
        self.busyChanged.emit(True)
        self._pool.start(worker)
        return task_id

    def cancel(self, task_id: str) -> bool:
        worker = self._workers.get(task_id)
        if worker is None:
            return False
        worker.token.cancel()
        return True

    def cancel_all(self) -> None:
        for worker in list(self._workers.values()):
            worker.token.cancel()

    def wait_for_idle(self, timeout_ms: int = 5000) -> bool:
        return bool(self._pool.waitForDone(int(timeout_ms)))

    def task_name(self, task_id: str) -> str:
        return self._names.get(task_id, "")

    @property
    def active_task_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._workers)

    @property
    def active_count(self) -> int:
        return len(self.active_task_ids)

    @property
    def is_busy(self) -> bool:
        return bool(self.active_task_ids)

    def _on_started(self, task_id: str) -> None:
        self.taskStarted.emit(task_id, self._names.get(task_id, ""))

    def _on_progress(self, task_id: str, value: int, message: str) -> None:
        self.taskProgress.emit(task_id, value, message)

    def _on_succeeded(self, task_id: str, result: object) -> None:
        self.taskSucceeded.emit(task_id, result)

    def _on_failed(self, task_id: str, message: str) -> None:
        self.taskFailed.emit(task_id, message)

    def _on_cancelled(self, task_id: str) -> None:
        self.taskCancelled.emit(task_id)

    def _on_finished(self, task_id: str) -> None:
        with self._lock:
            worker = self._workers.pop(task_id, None)
            self._names.pop(task_id, None)
        if worker is not None:
            worker.signals.deleteLater()
        self.taskFinished.emit(task_id)
        self.busyChanged.emit(self.is_busy)
