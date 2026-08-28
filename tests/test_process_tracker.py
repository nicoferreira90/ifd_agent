"""Unit tests for the IFD process tracker's state-bubbling logic."""

from ifd_agent.process_tracker import ProcessTracker


def _start_with_step(state: str) -> ProcessTracker:
    tracker = ProcessTracker(loan_id="loan-1", request_id="req-1")
    tracker.start_process("ifd")
    tracker.start_step("some-step", process_code="ifd")
    tracker.complete_step("some-step", process_code="ifd", state=state)
    return tracker


def test_complete_process_marks_completed_when_all_steps_completed() -> None:
    tracker = _start_with_step(state="completed")

    tracker.complete_process("ifd")

    proc = tracker.processes["ifd"]
    assert proc.execution_state_code == "completed"


def test_complete_process_marks_errored_when_any_step_errored() -> None:
    tracker = _start_with_step(state="errored")

    tracker.complete_process("ifd")

    proc = tracker.processes["ifd"]
    assert proc.execution_state_code == "errored"
    assert proc.notes is not None and "some-step" in proc.notes


def test_complete_process_marks_needs_review_when_any_step_needs_review() -> None:
    # Critical: a `needs_review` step must promote the process to
    # `needs_review`, NOT `errored`. This is what stops the upstream
    # queue worker from re-firing the same request after Haiku returns
    # low confidence.
    tracker = _start_with_step(state="needs_review")

    tracker.complete_process("ifd")

    proc = tracker.processes["ifd"]
    assert proc.execution_state_code == "needs_review"
    assert proc.notes is not None and "some-step" in proc.notes


def test_complete_process_prefers_errored_over_needs_review() -> None:
    # If both an `errored` and a `needs_review` step exist, the process
    # is `errored` — the harder failure wins.
    tracker = ProcessTracker(loan_id="loan-1", request_id="req-1")
    tracker.start_process("ifd")
    tracker.start_step("review-step", process_code="ifd")
    tracker.complete_step("review-step", process_code="ifd", state="needs_review")
    tracker.start_step("error-step", process_code="ifd")
    tracker.complete_step("error-step", process_code="ifd", state="errored")

    tracker.complete_process("ifd")

    proc = tracker.processes["ifd"]
    assert proc.execution_state_code == "errored"
