"""IFD Agent for AgentCore Runtime.

This agent handles Initial Flood Determination (IFD) automation on the FEMA
Map Service Center (msc.fema.gov), driving the portal with Playwright,
extracting flood zones from the captured PDF via a Bedrock vision
call, then writing the zone back to Encompass and uploading the PDF
to eFolder bucket 132.

Uses BedrockAgentCoreApp + strands Agent framework with hook-based loop guards.
"""

# comment for re-deployment... by Nico

import json
import os
import re
import uuid
from threading import Lock
from typing import Any

from af.tools import configure_cloudwatch_watchtower_logging, logger
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from dotenv import load_dotenv
from strands import Agent
from strands.hooks import (
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)

from ifd_agent.encompass_mcp_tool import (
    _process_request_called,
    get_encompass_loan_status,
    get_encompass_stats,
    get_property_address,
    process_encompass_request,
)
from ifd_agent.process_tracker import ProcessTracker, clear_tracker, set_tracker

load_dotenv()

app = BedrockAgentCoreApp()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
RUNTIME_NAME = os.getenv("RUNTIME_NAME", "ifd-agent")

_runtime_id = RUNTIME_NAME.replace("-", "_")
_log_group = f"/aws/bedrock-agentcore/runtimes/{_runtime_id}/app-logs"
_cw_handler: object = None
try:
    _cw_handler = configure_cloudwatch_watchtower_logging(
        log_group_name=_log_group,
        level="DEBUG",
        region=AWS_REGION,
        stream_id=uuid.uuid4().hex,
    )
except Exception:
    logger.warning("CloudWatch watchtower logging unavailable (missing credentials?)")


DEFAULT_MAX_TURNS = 3
MIN_MAX_TURNS = 1
MAX_MAX_TURNS = 100
PROCESS_TOOL_NAME = "process_encompass_request"


def _coerce_max_turns(raw_value: Any) -> int:
    """Normalize max_turns input into a safe integer range."""
    if not isinstance(raw_value, int) or isinstance(raw_value, bool):
        return DEFAULT_MAX_TURNS
    return max(MIN_MAX_TURNS, min(raw_value, MAX_MAX_TURNS))


def _extract_tool_result_text(result: Any) -> str | None:
    """Best-effort extraction of text content from a Strands tool result payload."""
    if isinstance(result, str) and result.strip():
        return result

    if isinstance(result, dict):
        text = result.get("text")
        if isinstance(text, str) and text.strip():
            return text

        content = result.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_text = block.get("text")
                    if isinstance(block_text, str) and block_text.strip():
                        return block_text
                elif isinstance(block, str) and block.strip():
                    return block

    return None


class IFDMaxTurnsExceeded(RuntimeError):
    """Raised when IFD exceeds its configured model-call budget."""


class _IFDLimitToolCounts(HookProvider):
    """Cancel repeated expensive tool invocations within a single request."""

    def __init__(self, max_tool_counts: dict[str, int]) -> None:
        self.max_tool_counts = max_tool_counts
        self.tool_counts: dict[str, int] = {}
        self.last_process_result: str | None = None
        self._lock = Lock()

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeInvocationEvent, lambda e: self.reset_counts(e))
        registry.add_callback(BeforeToolCallEvent, lambda e: self.intercept_tool(e))
        registry.add_callback(AfterToolCallEvent, lambda e: self.capture_process_result(e))

    def reset_counts(self, _event: BeforeInvocationEvent) -> None:
        with self._lock:
            self.tool_counts = {}
            self.last_process_result = None

    def intercept_tool(self, event: BeforeToolCallEvent) -> None:
        tool_use = getattr(event, "tool_use", {}) or {}
        tool_name = str(tool_use.get("name", "unknown"))

        with self._lock:
            max_count = self.max_tool_counts.get(tool_name)
            count = self.tool_counts.get(tool_name, 0) + 1
            self.tool_counts[tool_name] = count

        if max_count is not None and count > max_count:
            event.cancel_tool = (
                f"Tool '{tool_name}' has reached its invocation limit ({max_count}). "
                "Do NOT call this tool again. Use the existing tool result to finish "
                "the response."
            )

    def capture_process_result(self, event: AfterToolCallEvent) -> None:
        tool_use = getattr(event, "tool_use", {}) or {}
        tool_name = str(tool_use.get("name", "unknown"))
        if tool_name != PROCESS_TOOL_NAME:
            return

        result_text = _extract_tool_result_text(getattr(event, "result", None))
        if result_text:
            with self._lock:
                self.last_process_result = result_text


class _IFDMaxTurns(HookProvider):
    """Abort the agent when it exceeds the configured number of model turns."""

    def __init__(self, max_turns: int) -> None:
        self.max_turns = _coerce_max_turns(max_turns)
        self._turn_count = 0
        self._lock = Lock()

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeInvocationEvent, lambda e: self.reset_turns(e))
        registry.add_callback(BeforeModelCallEvent, lambda e: self.check_turn_count(e))

    def reset_turns(self, _event: BeforeInvocationEvent) -> None:
        with self._lock:
            self._turn_count = 0

    def check_turn_count(self, _event: BeforeModelCallEvent) -> None:
        with self._lock:
            self._turn_count += 1
            turn_count = self._turn_count

        if turn_count > self.max_turns:
            raise IFDMaxTurnsExceeded(f"IFD agent exceeded max_turns ({self.max_turns})")


def create_ifd_agent(
    max_turns: int = DEFAULT_MAX_TURNS,
) -> tuple[Agent, _IFDLimitToolCounts]:
    """Create the IFD agent with all tools registered.

    Handles Initial Flood Determination via the FEMA NFHL viewer (reached
    through msc.fema.gov), with vision-based zone extraction
    and Encompass write-back (fields 2365 + 2366 + 2367 + 541 + eFolder bucket 132).
    """
    system_prompt = """You are Moder, an AI assistant specialized in Initial Flood Determination (IFD) for Encompass.

Your capabilities include:
- Driving the FEMA Map Service Center (msc.fema.gov) to capture flood determination PDFs.
- Extracting the FEMA flood zone from the captured PDF with a Bedrock vision call.
- Writing the determination back to Encompass fields 2365 (Determination Date), 2366 (boolean), 2367 (FEMA zone dropdown), and 541 (Operations flood zone dropdown).
- Uploading FLOODSEARCH.pdf to Encompass eFolder bucket 132 - Flood Search.

The workflow:
1. Look up the subject property address on the Encompass loan.
2. Search FEMA Map Service Center for that address via Playwright browser automation.
3. Capture the rendered map page as FLOODSEARCH.pdf.
4. Extract the flood zone, SFHA status, panel info, and community ID from the PDF with Bedrock vision.
5. Write the Determination Date to Encompass field 2365, the in-flood-zone boolean to field 2366, and the zone code to dropdowns 2367 and 541.
6. Upload the PDF to eFolder bucket 132.

## TOOL INVOCATION RULES

When the user requests IFD processing, call `process_encompass_request` EXACTLY ONCE. It internally
handles property lookup, browser automation, PDF capture, vision extraction, field write, and
eFolder upload. After it returns, format the result into the response — do not call any tool again.

- Do NOT call `process_encompass_request` a second time for any reason.
- If the result contains errors, partial failures, or `needs_review`, report them as-is.
- If the user asks about loan status, call `get_encompass_loan_status`. For stats/metrics, call
  `get_encompass_stats`. For the subject property address, call `get_property_address`.
- When processing is requested, IMMEDIATELY call `process_encompass_request` — no preamble.
- If the user provides ONLY a loan number, assume IFD context.

Examples:
- "Process loan 87025103184 for IFD portal" -> IMMEDIATELY call process_encompass_request("Process loan 87025103184 for IFD portal")
- "Run flood determination for loan 12345" -> IMMEDIATELY call process_encompass_request("Run flood determination for loan 12345")
- "Initial flood determination on loan 67890" -> IMMEDIATELY call process_encompass_request("Initial flood determination on loan 67890")
- "87025103184" -> IMMEDIATELY call process_encompass_request("Process loan 87025103184 for IFD portal")

## FORMATTING REQUIREMENTS
- Use proper markdown formatting with headers, lists, and emphasis.
- Convert ALL URLs to clickable links using [text](url) format.
- Structure information in clear sections.
- Use code blocks for technical details and filenames.

## MANDATORY RESPONSE SECTIONS
After the single tool call returns, format the response with these sections:
1. **ENCOMPASS eFOLDER PROCESSING COMPLETE (IFD)** header
2. **Loan Details** with ID, portal, status, timestamp
3. **Subject Property** section with the address used for the FEMA search
4. **Flood Zone Extraction** section (zone, SFHA, panel, community)
5. **URLs REQUESTED** section with FEMA Map Service Center URL
6. **PDF DOWNLOAD URLs** section with FLOODSEARCH.pdf location
7. **Updated Encompass Fields** section listing the fields that were written (on success: 2365, 2366, 2367, 541; plus CX.INITIAL.FLOOD.DETER.* when the Initial Processing button is pressed)
8. Final status footer (SUCCESS / PARTIAL / NEEDS REVIEW / SKIPPED / ERROR)
"""

    normalized_max_turns = _coerce_max_turns(max_turns)
    limit_hook = _IFDLimitToolCounts({PROCESS_TOOL_NAME: 1})
    max_turns_hook = _IFDMaxTurns(normalized_max_turns)

    logger.info(f"Creating IFD Agent with MODEL_ID: {MODEL_ID}, max_turns={normalized_max_turns}")

    agent = Agent(
        model=MODEL_ID,
        system_prompt=system_prompt,
        name="IFDAgent",
        tools=[
            process_encompass_request,
            get_encompass_loan_status,
            get_encompass_stats,
            get_property_address,
        ],
    )
    agent.hooks.add_hook(limit_hook)
    agent.hooks.add_hook(max_turns_hook)
    return agent, limit_hook


def log_response_payload(response: dict[str, Any], is_error: bool = False) -> None:
    """Log response payload with structured truncation."""
    log_fn = logger.error if is_error else logger.info
    prefix = "ERROR" if is_error else "FINAL"

    log_fn(f"{prefix} RESPONSE - status: {response.get('status')}")
    log_fn(f"{prefix} RESPONSE - loan_id: {response.get('loan_id')}")
    log_fn(f"{prefix} RESPONSE - request_id: {response.get('request_id')}")

    if is_error and response.get("error"):
        log_fn(f"{prefix} RESPONSE - error: {response.get('error')}")

    processes = response.get("processes", [])
    for proc in processes:
        proc_state = proc.get("execution_state_code", "unknown")
        proc_notes = proc.get("notes", "")
        log_fn(f"{prefix} RESPONSE - process '{proc.get('code')}': {proc_state}")
        if proc_notes:
            log_fn(f"{prefix} RESPONSE - process notes: {proc_notes[:500]}")

        for step in proc.get("steps", []):
            step_state = step.get("execution_state_code", "unknown")
            step_notes = step.get("notes", "")
            log_fn(f"{prefix} RESPONSE -   step '{step.get('code')}': {step_state}")
            if step_notes:
                log_fn(
                    f"{prefix} RESPONSE -   step notes: {step_notes[:300]}"
                    f"{'...' if len(step_notes) > 300 else ''}"
                )

    human_response = response.get("response", "")
    if human_response:
        truncated = human_response[:500] + "..." if len(human_response) > 500 else human_response
        log_fn(f"{prefix} RESPONSE - response (truncated): {truncated}")


def _looks_like_timestamp(candidate: str) -> bool:
    """Check if a numeric string looks like a timestamp."""
    year_pattern = r"^(19|20)\d{2}"
    if re.match(year_pattern, candidate):
        date_pattern = r"^(19|20)\d{2}(0[1-9]|1[0-2])"
        if re.match(date_pattern, candidate):
            return True
        if len(candidate) >= 8:
            return True
    return False


def extract_loan_id_from_prompt(prompt: str) -> str:
    """Extract loan ID from user prompt.

    Supports optional "-dev" suffix for development environment loans.
    """
    prompt_lower = prompt.lower()

    context_patterns = [
        r"loan\s+(?:id\s+)?#?(\d{5,}(?:-dev)?)\b",
        r"loan\s+number\s+#?(\d{5,}(?:-dev)?)\b",
        r"process\s+(?:loan\s+)?#?(\d{5,}(?:-dev)?)\b",
        r"#(\d{5,}(?:-dev)?)\b",
    ]

    for pattern in context_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            return match.group(1).lower()

    standalone_pattern = r"(?<![.$\d])(\d{8,11}(?:-dev)?)(?![.$\d])"
    match = re.search(standalone_pattern, prompt)
    if match:
        candidate = match.group(1)
        numeric_part = candidate.replace("-dev", "")
        if not _looks_like_timestamp(numeric_part):
            return candidate.lower()

    SHORT_PROMPT_THRESHOLD = 20
    if len(prompt.strip()) <= SHORT_PROMPT_THRESHOLD:
        simple_match = re.search(r"(\d{5,}(?:-dev)?)", prompt, re.IGNORECASE)
        if simple_match:
            return simple_match.group(1).lower()

    return "unknown"


@app.entrypoint
async def invoke(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Main entrypoint for AgentCore runtime.

    Payload format:
        {
            "prompt": "user query here",
            "request_id": "optional_request_id",
            "user_id": "optional_user_id",
            "max_turns": "optional max model calls"
        }

    Returns:
        {
            "status": "success" | "error",
            "loan_id": "extracted loan id",
            "request_id": "request identifier",
            "processes": [...],
            "response": "agent response text"
        }
    """
    tracker = None
    request_id = None
    _process_request_called.set(False)

    try:
        if not payload:
            return {"status": "error", "error": "No payload provided"}

        query = payload.get("prompt", "")
        if not query:
            return {"status": "error", "error": "No prompt in payload"}

        request_id = payload.get("request_id") or str(uuid.uuid4())
        max_turns = _coerce_max_turns(payload.get("max_turns", DEFAULT_MAX_TURNS))

        loan_id = extract_loan_id_from_prompt(query)

        tracker = ProcessTracker(loan_id=loan_id, request_id=request_id)
        set_tracker(tracker)

        tracker.start_process("ifd")

        agent, limit_hook = create_ifd_agent(max_turns=max_turns)

        tracker.start_step("agent-execution")
        response: Any | None = None
        response_text: str | None = None
        try:
            response = agent(query)
        except IFDMaxTurnsExceeded as agent_err:
            if limit_hook.last_process_result:
                logger.warning(
                    "IFD max_turns exceeded after processing completed successfully: "
                    f"{agent_err}"
                )
                response_text = limit_hook.last_process_result
            else:
                raise
        except Exception as agent_err:
            ifd_process = tracker.processes.get("ifd")
            has_errored_step = ifd_process and any(
                s.execution_state_code == "errored" for s in ifd_process.steps
            )
            tool_executed = ifd_process and any(
                s.code == "get-property-address" and s.execution_state_code == "completed"
                for s in ifd_process.steps
            )
            if tool_executed and not has_errored_step:
                logger.warning(
                    f"SDK error after business logic completed, treating as success: {agent_err}"
                )
                response = None
                if limit_hook.last_process_result:
                    response_text = limit_hook.last_process_result
            else:
                raise

        if response_text is None:
            response_text = (
                response.message["content"][0]["text"]
                if response is not None
                else "Processing completed (SDK response unavailable)"
            )

        input_tokens = 0
        output_tokens = 0

        if response is not None and hasattr(response, "usage"):
            usage = response.usage
            logger.debug(f"Found response.usage: {usage}")
            if isinstance(usage, dict):
                input_val = usage.get("inputTokens", usage.get("input_tokens", 0))
                output_val = usage.get("outputTokens", usage.get("output_tokens", 0))
                input_tokens = int(input_val) if input_val is not None else 0
                output_tokens = int(output_val) if output_val is not None else 0
            elif hasattr(usage, "inputTokens"):
                input_tokens = getattr(usage, "inputTokens", 0)
                output_tokens = getattr(usage, "outputTokens", 0)

        if input_tokens == 0 and response is not None and hasattr(response, "metrics"):
            metrics = response.metrics
            logger.debug(f"Found response.metrics: {metrics}")

            if hasattr(metrics, "accumulated_usage"):
                accumulated = metrics.accumulated_usage
                logger.debug(f"Found metrics.accumulated_usage: {accumulated}")
                if isinstance(accumulated, dict):
                    input_val = accumulated.get("inputTokens", 0)
                    output_val = accumulated.get("outputTokens", 0)
                    input_tokens = int(input_val) if input_val is not None else 0
                    output_tokens = int(output_val) if output_val is not None else 0
            elif isinstance(metrics, dict):
                input_val = metrics.get("inputTokens", metrics.get("input_tokens", 0))
                output_val = metrics.get("outputTokens", metrics.get("output_tokens", 0))
                input_tokens = int(input_val) if input_val is not None else 0
                output_tokens = int(output_val) if output_val is not None else 0

        if input_tokens == 0 and response is not None and hasattr(response, "message"):
            msg = response.message
            if isinstance(msg, dict) and "usage" in msg:
                msg_usage = msg.get("usage")
                if isinstance(msg_usage, dict):
                    logger.debug(f"Found message.usage: {msg_usage}")
                    input_val = msg_usage.get("inputTokens", msg_usage.get("input_tokens", 0))
                    output_val = msg_usage.get("outputTokens", msg_usage.get("output_tokens", 0))
                    input_tokens = int(input_val) if input_val is not None else 0
                    output_tokens = int(output_val) if output_val is not None else 0

        logger.info(f"Token usage - input: {input_tokens}, output: {output_tokens}")

        tracker.complete_step("agent-execution")
        tracker.complete_process("ifd")

        payload_data = tracker.to_dict(input_tokens=input_tokens, output_tokens=output_tokens)

        has_errored_process = any(
            proc.get("execution_state_code") == "errored" for proc in payload_data["processes"]
        )
        has_needs_review_process = any(
            proc.get("execution_state_code") == "needs_review" for proc in payload_data["processes"]
        )

        # `needs_review` is a terminal outcome where the vision model finished cleanly
        # but couldn't confidently extract a zone — a human has to look at
        # the eFolder PDF. It is *not* an error and must not be retried.
        # Surface it as a distinct top-level status so the upstream queue
        # worker (which routes on top_level_status) treats it terminally.
        if has_errored_process:
            top_level_status = "error"
        elif has_needs_review_process:
            top_level_status = "needs_review"
        else:
            top_level_status = "success"

        # Markers in error notes that indicate the error is deterministic
        # given the current inputs — retrying won't change the outcome and
        # only burns compute + Bedrock calls. Anything not matched here
        # stays retryable so transient browser / network failures recover.
        non_retryable_markers = (
            "locked by",  # LoanLockConflictError — needs the other user to release
            "low_confidence",  # vision model couldn't extract from the captured PDF
            "zone_found=no",  # same — terminal until a human reviews
            # Encompass field-write 400s are deterministic — the API is
            # saying our payload is structurally wrong (e.g. wrong field ID,
            # or value not in dropdown allowed set). Retrying with the same
            # inputs hits the same error and burns a full FEMA capture +
            # Bedrock call. Stay terminal until code or config is fixed.
            "invalid field id",
            "bad request",
        )

        # `needs_review` is terminal by definition; `error` is retryable
        # unless the notes match a deterministic-failure marker.
        retryable = top_level_status == "error"
        if has_errored_process:
            errored_notes = " ".join(
                step.get("notes") or ""
                for proc in payload_data["processes"]
                for step in proc.get("steps", [])
                if step.get("execution_state_code") == "errored"
            )
            notes_lower = errored_notes.lower()
            if any(marker in notes_lower for marker in non_retryable_markers):
                retryable = False

        logger.info("=" * 60)
        logger.info("AGENT PROCESS TRACKER PAYLOAD")
        logger.info("=" * 60)
        logger.info(f"loan_id: {loan_id}")
        logger.info(f"request_id: {request_id}")
        logger.info(f"top_level_status: {top_level_status}")
        logger.info(f"retryable: {retryable}")
        logger.info(f"processes: {json.dumps(payload_data['processes'], indent=2)}")
        logger.info("=" * 60)

        final_response = {
            "status": top_level_status,
            "retryable": retryable,
            "loan_id": loan_id,
            "request_id": request_id,
            "input_tokens_used": input_tokens,
            "output_tokens_used": output_tokens,
            "processes": payload_data["processes"],
            "response": response_text,
        }

        log_response_payload(final_response, is_error=False)

        return final_response

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        logger.error(f"ERROR in invoke: {e}")
        logger.error(f"Traceback: {error_details}")

        if tracker:
            tracker.error_process("ifd", str(e))
            payload_data = tracker.to_dict(input_tokens=0, output_tokens=0)

            logger.error("=" * 60)
            logger.error("AGENT PROCESS TRACKER PAYLOAD (ERROR)")
            logger.error("=" * 60)
            logger.error(f"loan_id: {payload_data['loan_id']}")
            logger.error(f"request_id: {payload_data['request_id']}")
            logger.error(f"processes: {json.dumps(payload_data['processes'], indent=2)}")
            logger.error("=" * 60)

            error_response = {
                "status": "error",
                "loan_id": payload_data["loan_id"],
                "request_id": payload_data["request_id"],
                "input_tokens_used": 0,
                "output_tokens_used": 0,
                "processes": payload_data["processes"],
                "error": str(e),
                "traceback": error_details,
            }

            log_response_payload(error_response, is_error=True)

            return error_response

        fallback_loan_id = "unknown"
        if payload and payload.get("prompt"):
            extracted = extract_loan_id_from_prompt(payload.get("prompt", ""))
            if extracted:
                fallback_loan_id = extracted

        return {
            "status": "error",
            "loan_id": fallback_loan_id,
            "request_id": request_id,
            "input_tokens_used": 0,
            "output_tokens_used": 0,
            "processes": [],
            "error": str(e),
            "traceback": error_details,
        }

    finally:
        clear_tracker()


if __name__ == "__main__":
    app.run()
