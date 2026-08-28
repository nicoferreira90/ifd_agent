"""Encompass MCP Tool for IFD Agent.

This is an IFD-only agent that automates the Initial Flood Determination
pipeline. Handles natural-language requests by extracting a loan ID,
dispatching to the core ``process_loan`` flow, and emitting step-level
process-tracker events for the orchestrator UI.
"""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from af.tools import logger
from strands import tool

from ifd_agent.encompass_functions import (
    get_loan_property_address as get_loan_property_address_func,
)
from ifd_agent.encompass_functions import (
    get_loan_status_async,
    get_stats_async,
    process_loan,
)
from ifd_agent.process_tracker import get_tracker

_process_request_called: ContextVar[bool] = ContextVar("_process_request_called", default=False)

# Step name constants for the IFD process tracker.
STEP_GET_PROPERTY_ADDRESS = "get-property-address"
STEP_SEARCH_FEMA = "search-website-fema"
STEP_CAPTURE_PDF_FEMA = "capture-website-pdf-fema"
STEP_EXTRACT_FLOOD_ZONE = "extract-flood-zone-vision"
STEP_WRITE_FLOOD_ZONE_FIELD = "write-flood-zone-field"
STEP_UPLOAD_EFOLDER_IFD = "upload-to-encompass-efolder-132-flood"


class EncompassMCPTool:
    """MCP Tool for Encompass eFolder operations - IFD Agent.

    Handles natural-language requests for Initial Flood Determination.
    All functions are direct calls (no HTTP server needed).
    """

    def __init__(self) -> None:
        self.portals = ["IFD"]

    def process_request(self, user_input: str) -> dict[str, Any]:
        """Process natural-language request for Encompass IFD operations.

        Examples:
            - "process loan 12345 for IFD portal"
            - "run flood search for loan 67890"
            - "initial flood determination for loan 11111"
        """
        try:
            parsed_request = self._parse_user_input(user_input)

            if not parsed_request:
                return {
                    "success": False,
                    "error": "Could not understand the request. Please specify a loan ID.",
                    "suggestions": [
                        "Try: 'process loan 12345 for IFD portal'",
                        "Try: 'run flood search for loan 67890'",
                        "Try: 'initial flood determination for loan 11111'",
                    ],
                }

            result = self._process_loan(parsed_request["loan_id"], parsed_request["portal"])
            return result

        except Exception as e:
            logger.exception(f"Error processing request: {e}")
            return {"success": False, "error": f"Error processing request: {str(e)}"}

    def _parse_user_input(self, user_input: str) -> dict[str, str] | None:
        """Parse natural language input to extract loan_id and portal.

        Supports optional "-dev" suffix for development environment loans.
        """
        user_input_lower = user_input.lower()

        loan_id_patterns = [
            r"loan\s+(?:id\s+)?(\d+(?:-dev)?)",
            r"loan\s+number\s+(\d+(?:-dev)?)",
            r"(\d{5,}(?:-dev)?)",
            r"loan\s+(\d+(?:-dev)?)",
        ]

        loan_id = None
        for pattern in loan_id_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                loan_id = match.group(1)
                break

        if not loan_id:
            return None

        portal = "IFD"
        return {"loan_id": loan_id, "portal": portal}

    def _run_async_in_thread(self, coro: Any) -> Any:
        """Run async function in a thread to avoid event loop conflicts."""
        try:
            asyncio.get_running_loop()
            with ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            return asyncio.run(coro)

    def _process_loan(self, loan_id: str, portal: str) -> dict[str, Any]:
        """Process loan using direct function calls (no API server needed)."""
        tracker = get_tracker()

        try:
            logger.info(
                f"Processing loan directly (no API server): loan_id={loan_id}, portal={portal}"
            )

            if tracker:
                tracker.start_step(STEP_GET_PROPERTY_ADDRESS)

            result = self._run_async_in_thread(process_loan(loan_id, portal))
            end_time = datetime.now()

            status_lower = str(result.get("status", "")).lower()
            # NOTE: "needs_review" is intentionally NOT treated as success. The
            # needs-review workflow is deferred (BRAVO-135); IFD currently
            # surfaces uncertain/unusable maps as "error", which routes through
            # the failure path below and is rendered as Errored in the UI.
            is_success = status_lower in [
                "success",
                "skipped",
                "partial",
                "completed",
            ]

            if tracker:
                self._emit_step_events(tracker, result, status_lower, is_success)

            detailed_info = {
                "success": is_success,
                "loan_id": result["loan_id"],
                "portal": result["portal"],
                "status": result["status"],
                "message": result["message"],
                "timestamp": end_time.isoformat(),
                "borrowers_processed": result.get("borrowers_processed", []),
                "time_elapsed": "Processing completed",
                "urls_requested": [],
                "pdf_download_urls": [],
                "playwright_prompts": [],
                "updated_fields": [],
                "exclusion_reason": result.get("exclusion_reason"),
                "loan_status": result.get("loan_status"),
                "whats_up": result.get("whats_up"),
                "skip_reason": result.get("skip_reason"),
                "days_old": result.get("days_old"),
                "document_date": result.get("document_date"),
                "expiration_check": result.get("expiration_check"),
                "address": result.get("address"),
                "flood_zone_extraction": result.get("flood_zone_extraction"),
                "field_update": result.get("field_update"),
            }

            for borrower in result.get("borrowers_processed", []):
                if "search_url" in borrower:
                    detailed_info["urls_requested"].append(
                        {
                            "portal": result["portal"],
                            "borrower": borrower.get("borrower_name", "Unknown"),
                            "url": borrower["search_url"],
                            "summary": (
                                "Searched FEMA Map Service Center for "
                                f"{borrower.get('borrower_name', 'Unknown')}"
                            ),
                        }
                    )

                if "pdf_filename" in borrower:
                    pdf_info = {
                        "portal": result["portal"],
                        "borrower": borrower.get("borrower_name", "Unknown"),
                        "filename": borrower["pdf_filename"],
                        "note": "PDF uploaded to eFolder bucket 132 - Flood Search",
                    }
                    if borrower.get("pdf_path"):
                        pdf_info["download_url"] = borrower["pdf_path"]
                    detailed_info["pdf_download_urls"].append(pdf_info)

                if borrower.get("encompass_fields_updated"):
                    zone = (borrower.get("flood_zone") or "").strip()
                    in_zone = (
                        "in flood zone" if borrower.get("in_flood_zone") else "not in flood zone"
                    )
                    detailed_info["updated_fields"].append(
                        f"eFolder upload + fields 2366/2367/2365 ({in_zone}, zone {zone}) "
                        f"for {borrower.get('borrower_name', 'Unknown')}"
                    )

            return detailed_info

        except TimeoutError:
            if tracker:
                tracker.error_step(STEP_GET_PROPERTY_ADDRESS, "Request timed out")
            return {
                "success": False,
                "error": "Request timed out. The loan processing is taking longer than expected.",
                "suggestion": "Please check the loan status later using the monitoring interface.",
            }
        except Exception as e:
            if tracker:
                tracker.error_step(STEP_GET_PROPERTY_ADDRESS, str(e))
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    def _emit_step_events(
        self,
        tracker: Any,
        result: dict[str, Any],
        status_lower: str,
        is_success: bool,
    ) -> None:
        """Emit the IFD step sequence into the process tracker.

        The first step (``STEP_GET_PROPERTY_ADDRESS``) is started by the caller
        before ``process_loan`` runs; we close it out here based on whether the
        flow proceeded past the property-lookup stage.
        """
        if not is_success:
            borrowers = result.get("borrowers_processed", [])
            primary = borrowers[0] if borrowers else {}
            address = result.get("address") or {}
            has_address = bool(address.get("street")) and bool(address.get("city"))
            extraction = result.get("flood_zone_extraction") or {}

            # Uncertain / unusable FEMA map: we reached the FEMA stage but could
            # not produce a confident flood determination — a corrupted render,
            # no FEMA digital data, a blank/low-confidence read, or an extraction
            # failure. No PDF is filed and no determination is written; the run
            # is surfaced as Errored (red, retryable) so it stays visible and is
            # retried. Walk the steps we completed, then error the step we
            # stopped at. (The needs-review workflow is deferred — BRAVO-135.)
            uncertain_map = has_address and (
                primary.get("map_unreadable") or primary.get("no_usable_map") or bool(extraction)
            )
            if uncertain_map:
                map_captured = bool(
                    primary.get("map_captured") or primary.get("map_unreadable") or extraction
                )
                tracker.complete_step(STEP_GET_PROPERTY_ADDRESS)
                tracker.start_step(STEP_SEARCH_FEMA)
                tracker.complete_step(STEP_SEARCH_FEMA)
                tracker.start_step(STEP_CAPTURE_PDF_FEMA)
                if map_captured:
                    # A PDF was captured but no confident flood zone could be read.
                    tracker.complete_step(STEP_CAPTURE_PDF_FEMA)
                    tracker.start_step(STEP_EXTRACT_FLOOD_ZONE)
                    tracker.error_step(
                        STEP_EXTRACT_FLOOD_ZONE,
                        primary.get("status")
                        or result.get("message")
                        or "FEMA flood map could not be classified",
                    )
                else:
                    # Capture itself failed deterministically — no PDF produced.
                    tracker.error_step(
                        STEP_CAPTURE_PDF_FEMA,
                        primary.get("status")
                        or result.get("message")
                        or "FEMA flood map unavailable",
                    )
                return

            # Generic early failure (missing address, loan lock, unexpected error).
            tracker.error_step(
                STEP_GET_PROPERTY_ADDRESS,
                result.get("message", "Processing failed"),
            )
            return

        address = result.get("address") or {}
        has_address = bool(address.get("street")) and bool(address.get("city"))

        if has_address or status_lower in ("skipped",):
            tracker.complete_step(STEP_GET_PROPERTY_ADDRESS)
        else:
            tracker.error_step(
                STEP_GET_PROPERTY_ADDRESS,
                "Property address missing on loan",
            )
            return

        if status_lower == "skipped":
            return

        borrowers = result.get("borrowers_processed", [])
        if not borrowers:
            return

        primary = borrowers[0]

        # Uncertain / unusable maps (no FEMA data, blank/low-confidence read,
        # corrupted render, extraction failure) are surfaced as "error" and
        # handled in the failure path above — they never reach here. This path
        # only runs for confident determinations. (Needs-review deferred —
        # BRAVO-135.)
        extraction = result.get("flood_zone_extraction") or {}

        if primary.get("search_url"):
            tracker.start_step(STEP_SEARCH_FEMA)
            tracker.complete_step(STEP_SEARCH_FEMA)

        if primary.get("pdf_filename"):
            tracker.start_step(STEP_CAPTURE_PDF_FEMA)
            tracker.complete_step(STEP_CAPTURE_PDF_FEMA)

        upload_step_emitted = False
        if "efolder_uploaded" in primary:
            tracker.start_step(STEP_UPLOAD_EFOLDER_IFD)
            upload_step_emitted = True
            if primary.get("efolder_uploaded"):
                tracker.complete_step(STEP_UPLOAD_EFOLDER_IFD)
            else:
                tracker.error_step(
                    STEP_UPLOAD_EFOLDER_IFD,
                    primary.get("status") or "Upload failed",
                )

        if extraction:
            tracker.start_step(STEP_EXTRACT_FLOOD_ZONE)
            zone_value = extraction.get("flood_zone") or ""
            tracker.complete_step(
                STEP_EXTRACT_FLOOD_ZONE,
                notes=f"flood_zone={zone_value} sfha={extraction.get('sfha') or 'unclear'}",
            )

        field_update = result.get("field_update") or {}
        if field_update:
            tracker.start_step(STEP_WRITE_FLOOD_ZONE_FIELD)
            if field_update.get("success"):
                in_flood_zone = field_update.get("in_flood_zone")
                flood_zone = field_update.get("flood_zone") or ""
                field_ids = field_update.get("field_ids") or []
                tracker.complete_step(
                    STEP_WRITE_FLOOD_ZONE_FIELD,
                    notes=(
                        f"in_flood_zone={in_flood_zone} flood_zone='{flood_zone}' "
                        f"fields={field_ids}"
                    ),
                )
            else:
                tracker.error_step(
                    STEP_WRITE_FLOOD_ZONE_FIELD,
                    field_update.get("error") or "Field write failed",
                )

        if not upload_step_emitted:
            tracker.start_step(STEP_UPLOAD_EFOLDER_IFD)
            if primary.get("encompass_fields_updated"):
                tracker.complete_step(STEP_UPLOAD_EFOLDER_IFD)
            else:
                tracker.error_step(
                    STEP_UPLOAD_EFOLDER_IFD,
                    primary.get("status") or "Upload failed",
                )

    def _create_summary(self, result: dict[str, Any]) -> str:
        """Create a comprehensive human-readable summary of the processing result."""
        if not result.get("success", False):
            return (
                f"Failed to process loan {result.get('loan_id', 'Unknown')}: "
                f"{result.get('error', 'Unknown error')}"
            )

        summary_parts: list[str] = []

        summary_parts.append("**ENCOMPASS eFOLDER PROCESSING COMPLETE (IFD)**")
        summary_parts.append("=" * 50)

        summary_parts.append(f"**Loan ID:** {result['loan_id']}")
        summary_parts.append(f"**Portal:** {result['portal']} (Initial Flood Determination)")
        summary_parts.append(f"**Status:** {result['status'].upper()}")
        summary_parts.append(f"**Timestamp:** {result['timestamp']}")
        summary_parts.append(
            f"**Time Elapsed:** {result.get('time_elapsed', 'Processing completed')}"
        )
        summary_parts.append("")

        if result.get("exclusion_reason"):
            summary_parts.append("**ELIGIBILITY CHECK:**")
            summary_parts.append("  • **Status:** NOT ELIGIBLE")
            summary_parts.append(f"  • **Reason:** {result['exclusion_reason']}")
            if result.get("loan_status"):
                summary_parts.append(f"  • **Loan Status:** {result['loan_status']}")
            if result.get("whats_up"):
                summary_parts.append(f"  • **What's Up?:** {result['whats_up']}")
            summary_parts.append("")
        elif result.get("skip_reason") == "document_not_expired":
            summary_parts.append("**DOCUMENT EXPIRATION CHECK (30-Day Threshold):**")
            summary_parts.append("  • **Status:** SKIPPED - Document not expired")
            days_old = result.get("days_old", "unknown")
            doc_date = result.get("document_date", "N/A")
            summary_parts.append(f"  • **Document Age:** {days_old} days old (< 30 days threshold)")
            summary_parts.append(f"  • **Document Date:** {doc_date}")
            summary_parts.append("  • **Action:** Processing skipped - document is still valid")
            summary_parts.append("")
        else:
            summary_parts.append("**VALIDATION CHECKS:**")
            summary_parts.append("  • **Eligibility Check:** PASSED")
            if result.get("expiration_check"):
                exp_check = result.get("expiration_check")
                if exp_check and isinstance(exp_check, dict) and exp_check.get("has_document"):
                    days_old = exp_check.get("days_old", "unknown")
                    is_expired = exp_check.get("is_expired", True)
                    summary_parts.append(
                        f"  • **Document Expiration Check (30-day):** "
                        f"{'EXPIRED' if is_expired else 'NOT EXPIRED'} ({days_old} days old)"
                    )
                else:
                    summary_parts.append(
                        "  • **Document Expiration Check (30-day):** NO DOCUMENT FOUND - Proceeding"
                    )
            summary_parts.append("")

        if result.get("address"):
            addr = result["address"]
            summary_parts.append("**SUBJECT PROPERTY:**")
            address_line = addr.get("formatted") or (
                f"{addr.get('street', '')}, {addr.get('city', '')}, "
                f"{addr.get('state', '')} {addr.get('postal_code', '')}"
            )
            summary_parts.append(f"  • {address_line}")
            summary_parts.append("")

        if result.get("flood_zone_extraction"):
            extraction = result["flood_zone_extraction"]
            summary_parts.append("**FLOOD ZONE EXTRACTION (Bedrock vision):**")
            summary_parts.append(f"  • **Flood Zone:** {extraction.get('flood_zone') or 'N/A'}")
            summary_parts.append(f"  • **SFHA:** {extraction.get('sfha') or 'unclear'}")
            summary_parts.append(
                f"  • **Panel:** {extraction.get('panel_number') or 'N/A'} "
                f"(effective {extraction.get('panel_effective_date') or 'N/A'})"
            )
            summary_parts.append(
                f"  • **Community:** {extraction.get('community_name') or 'N/A'} "
                f"({extraction.get('community_id') or 'N/A'})"
            )
            summary_parts.append(f"  • **Zone Found:** {extraction.get('zone_found') or 'no'}")
            summary_parts.append("")

        if result.get("urls_requested"):
            summary_parts.append("**URLs REQUESTED:**")
            for url_info in result["urls_requested"]:
                summary_parts.append(f"  • **{url_info['portal']}** - {url_info['borrower']}")
                summary_parts.append(f"    URL: {url_info['url']}")
                summary_parts.append(f"    Summary: {url_info['summary']}")
                summary_parts.append("")

        if result.get("pdf_download_urls"):
            summary_parts.append("**PDF ARTIFACTS:**")
            for pdf_info in result["pdf_download_urls"]:
                summary_parts.append(f"  • **{pdf_info['portal']}** - {pdf_info['borrower']}")
                summary_parts.append(f"    Filename: {pdf_info['filename']}")
                if pdf_info.get("download_url"):
                    summary_parts.append(f"    Download: {pdf_info['download_url']}")
                if pdf_info.get("note"):
                    summary_parts.append(f"    Note: {pdf_info['note']}")
                summary_parts.append("")

        if result.get("updated_fields"):
            summary_parts.append("**UPDATED ENCOMPASS FIELDS:**")
            for field in result["updated_fields"]:
                summary_parts.append(f"  • {field}")
            summary_parts.append("")

        status_label_map = {
            "success": "**PROCESS COMPLETED SUCCESSFULLY**",
            "partial": "**PROCESS COMPLETED WITH PARTIAL SUCCESS**",
            "needs_review": "**PROCESS COMPLETED — NEEDS HUMAN REVIEW**",
            "skipped": "**PROCESS SKIPPED**",
        }
        summary_parts.append(
            status_label_map.get(result["status"], "**PROCESS COMPLETED WITH ERRORS**")
        )

        return "\n".join(summary_parts)

    def get_loan_status(self, loan_id: str) -> dict[str, Any]:
        """Get status of a processed loan."""
        try:
            result = self._run_async_in_thread(get_loan_status_async(loan_id))
            if not isinstance(result, dict):
                raise TypeError(
                    f"Expected dict from get_loan_status_async, got {type(result).__name__}"
                )
            return result
        except Exception as e:
            logger.exception(f"Error in get_loan_status: {e}")
            return {"success": False, "error": f"Error checking loan status: {str(e)}"}

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        try:
            result = self._run_async_in_thread(get_stats_async())
            if not isinstance(result, dict):
                raise TypeError(f"Expected dict from get_stats_async, got {type(result).__name__}")
            return result
        except Exception as e:
            logger.exception(f"Error in get_stats: {e}")
            return {"success": False, "error": f"Error fetching stats: {str(e)}"}


encompass_tool = EncompassMCPTool()


@tool
def process_encompass_request(user_input: str) -> str:
    """Process a loan for the Initial Flood Determination (IFD) portal.

    This tool processes a loan end-to-end by:
    1. Reading the subject property address from Encompass.
    2. Driving the FEMA Map Service Center (msc.fema.gov) with Playwright.
    3. Capturing a PDF of the flood determination map (FLOODSEARCH.pdf).
    4. Extracting the FEMA flood zone from the PDF with a Bedrock vision
       call.
    5. Writing the in-flood-zone boolean to Encompass field 2366 and (when in a zone) the zone code to dropdown 2367.
    6. Uploading the PDF to Encompass eFolder bucket 132 - Flood Search.

    Args:
        user_input: Natural-language request containing the loan ID.
            Examples:
            - "Process loan 12345 for IFD portal"
            - "Run flood determination for loan 67890"
            - "Initial flood determination for loan 11111"
            - "12345" (defaults to IFD)

    Returns:
        A formatted markdown string with processing results including the
        extracted flood zone, SFHA status, PDF location, and step-level state.
    """
    if _process_request_called.get():
        raise RuntimeError("process_encompass_request already called — do not retry")
    _process_request_called.set(True)

    result = encompass_tool.process_request(user_input)

    if result["success"]:
        return encompass_tool._create_summary(result)
    else:
        response = "**Encompass eFolder Processing Failed (IFD)**\n\n"
        response += f"**Error:** {result['error']}\n"

        if result.get("suggestions"):
            response += "\n**Suggestions:**\n"
            for suggestion in result["suggestions"]:
                response += f"• {suggestion}\n"

        if result.get("suggestion"):
            response += f"\n**Suggestion:** {result['suggestion']}"

        return response


@tool
def get_encompass_loan_status(loan_id: str) -> str:
    """Get the processing status of a specific loan.

    Args:
        loan_id: The loan ID to check status for (e.g., "87025103184").

    Returns:
        A formatted string with loan status, borrowers processed, and timestamps.
    """
    result = encompass_tool.get_loan_status(loan_id)

    if result["success"]:
        response = f"**Loan Status for {loan_id}**\n\n"
        response += f"**Status:** {result['status']}\n"
        response += f"**Message:** {result['message']}\n"
        response += f"**Timestamp:** {result['timestamp']}\n"

        if result.get("borrowers_processed"):
            response += "\n**Property:**\n"
            for borrower in result["borrowers_processed"]:
                response += (
                    f"• {borrower.get('borrower_name', 'Unknown')}: "
                    f"{borrower.get('status', 'Unknown')}\n"
                )
    else:
        response = f"**Error checking loan status**\n\n{result['error']}"

    return response


@tool
def get_encompass_stats() -> str:
    """Get overall processing statistics for all loans processed by this runtime.

    Returns:
        A formatted string with total loans, success rate, and processing statistics.
    """
    result = encompass_tool.get_stats()

    if result["success"]:
        stats = result["stats"]
        response = "**Encompass IFD Processing Statistics**\n\n"
        response += f"**Total Loans:** {stats['total_loans']}\n"
        response += f"**Successful:** {stats['successful_loans']}\n"
        response += f"**Failed:** {stats['failed_loans']}\n"
        response += f"**Success Rate:** {stats['success_rate']:.1f}%"
    else:
        response = f"**Error fetching statistics**\n\n{result['error']}"

    return response


@tool
def get_property_address(loan_id: str) -> str:
    """Get the subject property address for a specific loan.

    Args:
        loan_id: The loan ID to get property address for (e.g., "87025103184").

    Returns:
        A formatted string with street, city, state, and postal code for the
        subject property.
    """
    try:
        result = get_loan_property_address_func(loan_id)

        if "error" in result:
            return f"**Error getting property address**\n\n{result['error']}"

        address = result.get("address") or {}
        if address.get("street") or address.get("city"):
            response = f"**Subject Property for Loan {loan_id}:**\n\n"
            if address.get("street"):
                response += f"- Street: {address['street']}\n"
            if address.get("unit"):
                response += f"- Unit: {address['unit']}\n"
            if address.get("city"):
                response += f"- City: {address['city']}\n"
            if address.get("state"):
                response += f"- State: {address['state']}\n"
            if address.get("postal_code"):
                response += f"- Postal Code: {address['postal_code']}\n"
            if address.get("formatted"):
                response += f"\nOne-line: {address['formatted']}"
            return response
        else:
            return f"No subject property address found for loan {loan_id}"

    except Exception as e:
        logger.error(f"Error getting property address: {e}")
        return f"**Error getting property address**\n\n{str(e)}"
