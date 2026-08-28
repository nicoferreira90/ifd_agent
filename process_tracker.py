"""Process tracker for collecting step-by-step execution data.

Used to build structured AgentProcessPayload responses for the orchestrator.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ifd_agent.models import AgentProcessPayload


@dataclass
class StepData:
    """Data for a single processing step."""

    code: str
    execution_state_code: str = "pending"
    notes: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class ProcessData:
    """Data for a single process (e.g., IFD)."""

    code: str
    execution_state_code: str = "pending"
    notes: str | None = None
    steps: list[StepData] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ProcessTracker:
    """Tracks process execution steps for building AgentProcessPayload."""

    def __init__(self, loan_id: str, request_id: str):
        self.loan_id = loan_id
        self.request_id = request_id
        self.processes: dict[str, ProcessData] = {}
        self._current_process: str | None = None
        self.created_at = datetime.now(UTC)

    def start_process(self, process_code: str, notes: str | None = None) -> None:
        """Start tracking a new process."""
        self.processes[process_code] = ProcessData(
            code=process_code,
            execution_state_code="executing",
            notes=notes,
            started_at=datetime.now(UTC),
        )
        self._current_process = process_code

    def complete_process(
        self, process_code: str, state: str = "completed", notes: str | None = None
    ) -> None:
        """Mark a process as complete, errored, or needs-review.

        Step states bubble up to the process state in this order of severity:
        1. If any step is ``errored``, the process is ``errored``.
        2. Otherwise, if any step is ``needs_review``, the process is
           ``needs_review`` — terminal outcome that requires a human to look
           at the eFolder, but distinct from an outright error and *not*
           retryable. This is what stops the upstream queue worker from
           re-firing requests after the vision model returns low confidence.
        3. Otherwise, the process gets the requested ``state`` (default
           ``completed``).
        """
        if process_code in self.processes:
            process = self.processes[process_code]

            has_errored_step = any(step.execution_state_code == "errored" for step in process.steps)
            has_needs_review_step = any(
                step.execution_state_code == "needs_review" for step in process.steps
            )

            if has_errored_step:
                process.execution_state_code = "errored"
                if not notes:
                    errored_steps = [
                        s.code for s in process.steps if s.execution_state_code == "errored"
                    ]
                    process.notes = (
                        f"Process errored due to failed steps: {', '.join(errored_steps)}"
                    )
                else:
                    process.notes = notes
            elif has_needs_review_step:
                process.execution_state_code = "needs_review"
                if not notes:
                    review_steps = [
                        s.code for s in process.steps if s.execution_state_code == "needs_review"
                    ]
                    process.notes = f"Process needs review due to steps: {', '.join(review_steps)}"
                else:
                    process.notes = notes
            else:
                process.execution_state_code = state
                if notes:
                    process.notes = notes

            process.completed_at = datetime.now(UTC)

    def error_process(self, process_code: str, error_message: str) -> None:
        """Mark a process as errored."""
        self.complete_process(process_code, state="errored", notes=error_message)

    def start_step(self, step_code: str, process_code: str | None = None) -> None:
        """Start tracking a step within a process."""
        proc_code = process_code or self._current_process
        if not proc_code or proc_code not in self.processes:
            return

        step = StepData(
            code=step_code, execution_state_code="executing", started_at=datetime.now(UTC)
        )
        self.processes[proc_code].steps.append(step)

    def complete_step(
        self,
        step_code: str,
        process_code: str | None = None,
        state: str = "completed",
        notes: str | None = None,
    ) -> None:
        """Complete a step."""
        proc_code = process_code or self._current_process
        if not proc_code or proc_code not in self.processes:
            return

        for step in self.processes[proc_code].steps:
            if step.code == step_code and step.execution_state_code == "executing":
                step.execution_state_code = state
                step.completed_at = datetime.now(UTC)
                if notes:
                    step.notes = notes
                break

    def error_step(
        self, step_code: str, error_message: str, process_code: str | None = None
    ) -> None:
        """Mark a step as errored."""
        self.complete_step(
            step_code=step_code, process_code=process_code, state="errored", notes=error_message
        )

    def to_dict(self, input_tokens: int = 0, output_tokens: int = 0) -> dict[str, Any]:
        """Convert tracker data to dictionary for AgentProcessPayload."""
        processes = []
        for process in self.processes.values():
            steps = []
            for step in process.steps:
                steps.append(
                    {
                        "code": step.code,
                        "execution_state_code": step.execution_state_code,
                        "notes": step.notes,
                    }
                )

            processes.append(
                {
                    "code": process.code,
                    "execution_state_code": process.execution_state_code,
                    "notes": process.notes,
                    "steps": steps,
                }
            )

        return {
            "loan_id": self.loan_id,
            "request_id": self.request_id,
            "input_tokens_used": input_tokens,
            "output_tokens_used": output_tokens,
            "processes": processes,
        }

    def to_payload(self) -> "AgentProcessPayload":
        """Convert to AgentProcessPayload model."""
        return AgentProcessPayload(**self.to_dict())


_current_tracker: ProcessTracker | None = None


def get_tracker() -> ProcessTracker | None:
    """Get the current request's tracker."""
    global _current_tracker
    return _current_tracker


def set_tracker(tracker: ProcessTracker) -> None:
    """Set the current request's tracker."""
    global _current_tracker
    _current_tracker = tracker


def clear_tracker() -> None:
    """Clear the current tracker."""
    global _current_tracker
    _current_tracker = None
