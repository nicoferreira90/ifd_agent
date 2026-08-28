"""Pydantic models for agent response payloads.

These models define the structured response format that the IFD agent
returns to the orchestrator.
"""

from typing import Any

from pydantic import BaseModel, Field


class ProcessStep(BaseModel):
    """A single step within a process execution."""

    code: str = Field(
        ...,
        description="Step identifier (e.g., 'capture-website-pdf-fema', 'extract-flood-zone-vision')",
    )
    execution_state_code: str = Field(
        ..., description="Status: 'pending', 'paused', 'executing', 'errored', 'complete'"
    )
    notes: str | None = Field(default=None, description="Additional notes or error messages")


class Process(BaseModel):
    """A process execution (e.g., IFD)."""

    code: str = Field(..., description="Process identifier (e.g., 'ifd')")
    execution_state_code: str = Field(
        ..., description="Status: 'pending', 'paused', 'executing', 'errored', 'complete'"
    )
    notes: str | None = Field(default=None, description="Additional notes or error messages")
    steps: list[ProcessStep] = Field(
        default_factory=list, description="Steps executed within this process"
    )


class AgentProcessPayload(BaseModel):
    """Structured response payload returned to the orchestrator."""

    loan_id: str = Field(..., description="The loan ID being processed")
    request_id: str = Field(..., description="Unique request identifier")
    input_tokens_used: int = Field(
        default=0, ge=0, description="Total input tokens consumed by agent"
    )
    output_tokens_used: int = Field(
        default=0, ge=0, description="Total output tokens consumed by agent"
    )
    processes: list[Process] = Field(default_factory=list, description="List of processes executed")

    total: int | None = Field(default=None, description="Total number of results")
    limit: int | None = Field(default=None, description="Maximum results per page")
    offset: int | None = Field(default=None, description="Number of results skipped")
    has_more: bool | None = Field(default=None, description="Whether more results are available")

    def has_errors(self) -> bool:
        """Check if any process or step has errored."""
        for process in self.processes:
            if process.execution_state_code == "errored":
                return True
            for step in process.steps:
                if step.execution_state_code == "errored":
                    return True
        return False

    def to_response(self) -> dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return self.model_dump(exclude_none=True)
