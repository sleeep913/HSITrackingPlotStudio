import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hsi_plotter.tasks.worker import CancelToken, TaskCancelled, TaskRunner  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_until_idle(runner: TaskRunner, app: QApplication, *, timeout_ms: int = 10000) -> None:
    deadline = time.monotonic() + timeout_ms / 1000
    while runner.is_busy and time.monotonic() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)


def test_successful_task_returns_result(app: QApplication) -> None:
    runner = TaskRunner()
    results: list[object] = []
    runner.taskSucceeded.connect(lambda _task_id, result: results.append(result))
    finished: list[str] = []
    runner.taskFinished.connect(finished.append)

    runner.submit("sum", lambda reporter, token: 42)
    _wait_until_idle(runner, app)

    assert results == [42]
    assert len(finished) == 1
    assert not runner.is_busy


def test_progress_updates_are_reported_in_order(app: QApplication) -> None:
    runner = TaskRunner()
    seen: list[tuple[int, str]] = []
    runner.taskProgress.connect(lambda _task_id, value, message: seen.append((value, message)))

    def handler(reporter: object, token: CancelToken) -> str:
        reporter.report(1, 3, "first")  # type: ignore[attr-defined]
        reporter.report(2, 3, "second")  # type: ignore[attr-defined]
        reporter.report(3, 3, "third")  # type: ignore[attr-defined]
        return "done"

    runner.submit("scan", handler)
    _wait_until_idle(runner, app)

    assert seen == [(33, "first"), (67, "second"), (100, "third")]


def test_failing_task_reports_message_without_crashing(app: QApplication) -> None:
    runner = TaskRunner()
    failures: list[str] = []
    runner.taskFailed.connect(lambda _task_id, message: failures.append(message))

    def handler(reporter: object, token: CancelToken) -> None:
        raise ValueError("目录不可访问")

    runner.submit("scan", handler)
    _wait_until_idle(runner, app)

    assert failures == ["目录不可访问"]
    assert not runner.is_busy


def test_running_task_can_be_cancelled(app: QApplication) -> None:
    runner = TaskRunner()
    cancelled: list[str] = []
    runner.taskCancelled.connect(cancelled.append)

    def handler(reporter: object, token: CancelToken) -> str:
        while token.check():
            time.sleep(0.01)
        token.raise_if_cancelled()
        return "never reached"

    task_id = runner.submit("scan", handler)
    QTimer.singleShot(0, lambda: runner.cancel(task_id))
    _wait_until_idle(runner, app)

    assert cancelled == [task_id]
    assert runner.active_task_ids == ()


def test_cancel_token_raises_when_requested() -> None:
    token = CancelToken()

    assert token.check()
    token.cancel()
    assert not token.check()
    with pytest.raises(TaskCancelled):
        token.raise_if_cancelled()


def test_busy_state_is_reported_while_running(app: QApplication) -> None:
    runner = TaskRunner()
    states: list[bool] = []
    runner.busyChanged.connect(states.append)

    runner.submit("slow", lambda reporter, token: time.sleep(0.05))
    _wait_until_idle(runner, app)

    assert states == [True, False]
    assert runner.wait_for_idle(1000)
