"""Encompass functions for the IFD agent.

Single-property flow (Job Aid pp. 67-70):
1. Pull secrets and connect to Encompass.
2. Check loan eligibility and document expiration on bucket 132.
3. Read the subject property address from `loan.property`.
4. Drive the FEMA NFHL Web AppViewer with Playwright to capture a PDF of the
   property's flood map.
5. Hand the PDF to the vision extractor to pull the flood zone.
   If FEMA has no digital flood data for the property, write "No digital data
   available" to Initial Processing Comments (CX.INITPROCNOTES), skip the PDF
   upload and determination writes, and leave the loan open for manual review.
6. Otherwise write the determination to Encompass: the 2366 boolean "in flood
   zone" checkbox plus the 2367 and 541 FEMA Flood Zone dropdowns (the zone
   code when in a mapped zone, or "X" — the minimal-hazard designation — when
   not). Upload the PDF to eFolder bucket 132.
7. Press the "Initial Flood Determination" Initial Processing button (write its
   user/date/time completion fields) once both the determination write and the
   upload have succeeded — the final step that marks the loan complete.
"""

import os
import re
from datetime import datetime
from typing import Any

from af.tools import logger
from af.tools.secrets import get_secret_value
from af.utils.time_utils import client_now, encompass_timestamp

from ifd_agent.encompass_assistant.exp_apis import (
    check_document_expiration,
    get_custom_field_value,
    get_loan_status_and_whats_up,
    update_custom_fields,
)
from ifd_agent.encompass_assistant.get_connection import connection
from ifd_agent.encompass_assistant.get_property_address import get_property_address
from ifd_agent.encompass_assistant.upload_file import upload_file_into_efolder
from ifd_agent.exceptions import LoanLockConflictError
from ifd_agent.playwright_assistant.capturePDF_async import (
    NFHL_VIEWER_URL,
    capture_fema_pdf_async,
)
from ifd_agent.utils.misc import cleanup_pdf_storage
from ifd_agent.utils.s3_operation import upload_into_s3
from ifd_agent.vision_assistant.flood_zone_extractor import (
    FloodZoneExtraction,
    extract_flood_zone,
)

# Encompass field IDs used by the IFD pipeline. Promote to a constants module
# once we wire panel/community fields (open question in the spec).
#
# 2367 — legacy "FEMA Flood Zone" dropdown the operator UI exposes, adjacent to
#        the 2366 in-flood-zone checkbox. Accepts the standard FEMA codes
#        (X, AE, A, VE, AO, AH, X500, etc.). Always written: the zone code
#        when in a mapped zone, or "X" (the minimal-hazard / not-in-SFHA
#        designation) when not — never left blank. Writing unconditionally
#        also overwrites any stale code from a prior run.
# 541  — "FEMA Flood Zone" dropdown field Operations checks in Encompass.
#        Written with the same normalized value as 2367 so both surfaces stay
#        consistent while preserving the existing 2367 integration behavior.
# 2366 — "The property has been determined to be in a flood zone" boolean
#        checkbox. Always written: True when the property is in a zone (teal/
#        orange shading on the NFHL map) and False when it is not (unshaded).
#        Despite the operator UI showing it as Y/N, the Encompass API expects
#        a JSON boolean — sending the string "Y"/"N" fails with a
#        Serialization error (the field's underlying type is boolean, not a
#        Y/N enum). Matches the GSA agent's VASUMM.X85 boolean pattern.
# 2365 — "Determination Date". Written with the timestamp whenever the IFD
#        workflow reaches a successful determination (in-zone or not-in-zone).
#        Not written for errored, skipped, or needs-review runs.
FLOOD_ZONE_DROPDOWN_FIELD_ID = "2367"
OPERATIONS_FLOOD_ZONE_DROPDOWN_FIELD_ID = "541"
IN_FLOOD_ZONE_CHECKBOX_FIELD_ID = "2366"
DETERMINATION_DATE_FIELD_ID = "2365"

# "Initial Flood Determination" Initial Processing button (Job Aid pp. 67-70,
# final step). Pressing the button in Encompass is done by writing its
# completion fields — the same field-write mechanism the other Initial
# Processor agents use (e.g. GSA's CX.INITIALPROC.09/.10) — rather than
# driving the Forms > Loan Status > Initial Processing UI. Unlike those agents,
# which stamp one combined "MM/DD/YYYY HH:MM:SS AM/PM" field, the flood button
# splits the stamp across separate date and time fields:
#   .DATE — "MM/DD/YYYY"
#   .TIME — "HH:MM:SS AM/PM"
#   .USER — the Encompass username that performed the determination
INITIAL_PROC_FLOOD_DATE_FIELD_ID = "CX.INITIAL.FLOOD.DETER.DATE"
INITIAL_PROC_FLOOD_TIME_FIELD_ID = "CX.INITIAL.FLOOD.DETER.TIME"
INITIAL_PROC_FLOOD_USER_FIELD_ID = "CX.INITIAL.FLOOD.DETER.USER"

# CX.INITPROCNOTES — "Initial Processing Comments" free-text field (same field
# the Prior Note / SECI agents write). For IFD we write it in every terminal
# "no usable map" case — FEMA has no digital data, the map render failed/was
# blank, or the address couldn't be located (deterministic capture failure).
# In all of those we record the note, skip the PDF upload and the
# determination-field writes, and leave the loan open for manual review.
INITIAL_PROCESSING_COMMENTS_FIELD_ID = "CX.INITPROCNOTES"
NO_DIGITAL_DATA_COMMENT = "No digital data available"
# Base note for a corrupted/unreadable FEMA map render. Written to Initial
# Processing Comments prefixed with the run date (e.g. "6/18 Image not printing
# for flood map on FEMA site") so ops can see when the failed render occurred.
MAP_UNREADABLE_COMMENT_BASE = "Image not printing for flood map on FEMA site"
# Matches the exact dated form IFD writes for a map-unreadable comment (e.g.
# "6/18 Image not printing for flood map on FEMA site"). Used to recognize a
# stale IFD comment for clearing — see `_is_ifd_authored_ip_comment`.
_MAP_UNREADABLE_COMMENT_PATTERN = re.compile(
    r"^\d{1,2}/\d{1,2} " + re.escape(MAP_UNREADABLE_COMMENT_BASE) + r"$"
)

IFD_BUCKET_NAME = "132 - Flood Search"
IFD_PDF_FILENAME = "FLOODSEARCH.pdf"
IFD_EFOLDER_PORTAL = "IFD"

secrets_cache: dict[str, Any] | None = None

processed_loans_storage: dict[str, list[dict[str, Any]]] = {}


def get_cached_secrets() -> dict[str, Any]:
    """Get secrets from AWS Secrets Manager using centralized af-tools."""
    global secrets_cache
    if secrets_cache is None:
        platform_arn = os.environ.get("PLATFORM_SECRETS_ARN")
        if not platform_arn:
            raise ValueError("PLATFORM_SECRETS_ARN environment variable is required")
        try:
            secrets_cache = get_secret_value(platform_arn)
            logger.info("Using secrets from AWS Secrets Manager via af-tools")
        except Exception as e:
            logger.error(f"Error fetching secrets: {e}")
            raise RuntimeError(f"Failed to load secrets: {e}") from e

    if secrets_cache is None:
        raise RuntimeError("Secrets cache is unexpectedly None")
    if not isinstance(secrets_cache, dict):
        raise ValueError(f"Secrets cache is not a dict, got {type(secrets_cache)}")
    return secrets_cache


# Lowercased substrings that mark a TRANSIENT capture failure — one a retry
# could plausibly fix (network blip, slow load, browser hiccup). A capture
# exception matching any of these stays a retryable `error`; everything else
# from the capture stage is treated as a DETERMINISTIC "no usable map for this
# address" outcome and routed to manual review, so we never loop forever on a
# property the automation simply can't map.
_TRANSIENT_CAPTURE_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "net::err",
    "err_connection",
    "err_network",
    "err_internet",
    "connection reset",
    "connection refused",
    "target closed",
    "browser has been closed",
    "browser closed",
    "page crashed",
    "websocket",
    "econnreset",
)


def _is_transient_capture_error(exc: BaseException) -> bool:
    """Best-effort: True when a capture failure looks transient (retry may help).

    Recognized infra/timeout failures stay retryable so a one-off network or
    browser hiccup can self-heal on the next orchestrator pass. Anything not
    recognized is treated as a deterministic "no usable map" outcome and routed
    to manual review — that's the safer default because it stops the retry loop
    on properties the automation deterministically can't map.
    """
    message = str(exc).lower()
    if any(marker in message for marker in _TRANSIENT_CAPTURE_ERROR_MARKERS):
        return True
    return isinstance(exc, TimeoutError)


def _write_ip_comment(
    access_token: str, api_server: str, loan_guid: str, loan_id: str, comment: str
) -> None:
    """Write a note to Initial Processing Comments (CX.INITPROCNOTES).

    Best-effort for ordinary failures: a failed write is logged but not raised,
    because the comment is advisory and the loan still routes to its terminal
    outcome (manual review or error) regardless of whether the note landed.

    A ``LoanLockConflictError`` IS re-raised, however, so it propagates to
    ``process_loan``'s lock handler and the run returns the NON-retryable lock
    result. Swallowing it would let the map_unreadable path (which returns a
    retryable ``status="error"``) misclassify a lock conflict as a retryable
    failure and loop on the retry cron until the lock clears.
    """
    try:
        update_custom_fields(
            access_token,
            api_server,
            loan_guid,
            loan_id,
            [{"id": INITIAL_PROCESSING_COMMENTS_FIELD_ID, "value": comment}],
        )
        logger.info(
            f"ifd_ip_comment_written loan_id={loan_id} "
            f"field_id={INITIAL_PROCESSING_COMMENTS_FIELD_ID} "
            f"comment='{comment}'"
        )
    except LoanLockConflictError:
        raise
    except Exception as comment_error:
        logger.error(
            f"Failed to write Initial Processing Comments for loan {loan_id}: {comment_error}"
        )


def _write_no_data_ip_comment(
    access_token: str, api_server: str, loan_guid: str, loan_id: str
) -> None:
    """Write "No digital data available" to Initial Processing Comments.

    Used for every terminal "no usable map" outcome (FEMA no-data, blank/failed
    render, unlocatable address, deterministic capture failure).
    """
    _write_ip_comment(access_token, api_server, loan_guid, loan_id, NO_DIGITAL_DATA_COMMENT)


def _is_ifd_authored_ip_comment(value: str) -> bool:
    """True only for an EXACT match to one of IFD's own comment templates.

    Deliberately exact, not substring: if a human or another agent appended
    text to an IFD comment (or vice versa), the combined value no longer
    equals either template and is correctly treated as NOT IFD-authored, so
    it is never cleared.
    """
    return value == NO_DIGITAL_DATA_COMMENT or bool(_MAP_UNREADABLE_COMMENT_PATTERN.match(value))


def _clear_ip_comment_if_ifd_authored(
    access_token: str, api_server: str, loan_guid: str, loan_id: str
) -> None:
    """Clear a stale IFD comment from Initial Processing Comments after success.

    CX.INITPROCNOTES is a shared, ops-facing field also written by the Prior
    Note, SECI, and OFAC agents (and human processors) for their own unrelated
    concerns, and every writer overwrites the field wholesale. So this must
    NOT blindly blank the field just because IFD succeeded — it only clears
    it when the *current* value is an EXACT match for one of IFD's own comment
    templates (see `_is_ifd_authored_ip_comment`), written on an earlier failed
    attempt for this same loan. Anything else — including an empty field, or a
    note some other agent/human wrote (even one that merely mentions IFD's
    wording) — is left untouched.

    Known limitation: the read and the clear-write are two separate API calls
    with no read-then-write concurrency control on this shared field (Encompass
    exposes no version/ETag on fieldWriter for it). If another writer updates
    CX.INITPROCNOTES in the narrow window between our read and our write, that
    update could be overwritten. Requiring an exact template match (rather
    than a substring) makes this negligible in practice — it would need a
    write to land in that same short window AND still equal, byte-for-byte,
    one of the two fixed strings above.

    Best-effort and never raises: this runs only after the determination has
    already succeeded, so a failure to read or clear the comment must not
    change the run's outcome.
    """
    try:
        current_value = get_custom_field_value(
            access_token,
            api_server,
            loan_guid,
            loan_id,
            INITIAL_PROCESSING_COMMENTS_FIELD_ID,
        )
        if not current_value or not _is_ifd_authored_ip_comment(current_value):
            return
        update_custom_fields(
            access_token,
            api_server,
            loan_guid,
            loan_id,
            [{"id": INITIAL_PROCESSING_COMMENTS_FIELD_ID, "value": ""}],
        )
        # Deliberately omit the comment text: CX.INITPROCNOTES is a shared,
        # ops-facing free-text field that can carry human-entered notes, so
        # only the fact and length of the clear are logged, not its content.
        logger.info(
            f"ifd_ip_comment_cleared loan_id={loan_id} "
            f"field_id={INITIAL_PROCESSING_COMMENTS_FIELD_ID} "
            f"previous_value_length={len(current_value)}"
        )
    except Exception as clear_error:
        logger.warning(
            f"Failed to clear stale Initial Processing Comments for loan {loan_id}: {clear_error}"
        )


def _is_sfha_zone(zone: str) -> bool:
    """Return whether a FEMA flood-zone code denotes a Special Flood Hazard Area.

    SFHA ("in a flood zone") zones are exactly the A* and V* family — A, AE, AH,
    AO, AR, A99, A1-A30, V, VE, V1-V30. Everything else is NOT an SFHA: X (both
    the unshaded minimal-hazard zone AND the shaded/orange 0.2%-annual-chance
    zone), B, C, D, and an empty/unknown code. Orange/0.2% areas are Zone X and,
    per Operations, are NOT in a flood zone.

    SFHA status is a deterministic function of the zone code, so we classify it
    here rather than using ``bool(zone)`` (which wrongly treats a named non-SFHA
    zone like X or D as in-zone) or trusting the vision model's separate ``sfha``
    guess. This is the sole source of the 2366 in-flood-zone boolean.
    """
    normalized = zone.strip().upper()
    return bool(normalized) and normalized[0] in ("A", "V")


def _no_usable_map_result(
    *,
    loan_id: str,
    portal: str,
    address: dict[str, Any],
    ifd_expiration: dict[str, Any] | None,
    map_captured: bool,
    extraction_log: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the errored result for a terminal "no usable map" outcome.

    ``map_captured`` distinguishes the case where a PDF WAS captured but
    deliberately not filed (FEMA no-data / blank render that vision flagged)
    from a deterministic capture failure where no PDF exists at all. The
    borrower carries ``no_usable_map=True`` so the step emitter errors the
    stopping point and skips the (intentionally) missing upload step. We
    deliberately do NOT set ``pdf_filename`` here even when a PDF was captured
    — that key signals "PDF filed to eFolder" to downstream summary/UI code,
    and in these cases nothing was filed. ``map_captured`` carries the
    captured-but-not-filed distinction instead.

    NOTE: this surfaces as ``error`` (not ``needs_review``). The needs-review
    workflow is deferred (BRAVO-135); until it lands, uncertain/unusable maps
    are surfaced as Errored so they stay visible and are retried.
    """
    borrower: dict[str, Any] = {
        "borrower_name": address.get("formatted")
        or f"{address.get('street', '')}, {address.get('city', '')}, {address.get('state', '')}",
        "state": address.get("state", ""),
        "search_url": NFHL_VIEWER_URL,
        "status": NO_DIGITAL_DATA_COMMENT,
        "efolder_uploaded": False,
        "encompass_fields_updated": False,
        "no_usable_map": True,
        "map_captured": map_captured,
    }

    result: dict[str, Any] = {
        "loan_id": loan_id,
        "portal": portal,
        "status": "error",
        "message": (
            "No usable FEMA flood map for this property. No PDF was filed and no "
            "determination was written; run surfaced as errored for retry/visibility."
        ),
        "timestamp": datetime.now().isoformat(),
        "borrowers_processed": [borrower],
        "address": address,
        "expiration_check": ifd_expiration,
    }
    if extraction_log is not None:
        result["flood_zone_extraction"] = extraction_log
    return result


def _map_unreadable_result(
    *,
    loan_id: str,
    portal: str,
    address: dict[str, Any],
    ifd_expiration: dict[str, Any] | None,
    comment: str,
    extraction_log: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ERRORED result for a corrupted/unreadable FEMA map render.

    Distinct from ``_no_usable_map_result`` (which is a terminal needs_review
    outcome): here the map rendered but is too corrupted to classify, so we
    write only the Initial Processing Comments note and surface the run as an
    error. The borrower carries ``map_unreadable=True`` so the step emitter
    marks flood-zone extraction ``errored`` (top-level status → error). We do
    NOT write 2365/2366/2367/541, do NOT press the Initial Processing button, and
    do NOT file a PDF (no ``pdf_filename``). ``status="error"`` keeps the run
    retryable per the upstream queue worker, so a fresh capture can re-attempt
    a clean render on the next pass.
    """
    borrower: dict[str, Any] = {
        "borrower_name": address.get("formatted")
        or f"{address.get('street', '')}, {address.get('city', '')}, {address.get('state', '')}",
        "state": address.get("state", ""),
        "search_url": NFHL_VIEWER_URL,
        "status": comment,
        "efolder_uploaded": False,
        "encompass_fields_updated": False,
        "map_unreadable": True,
        "map_captured": True,
    }

    result: dict[str, Any] = {
        "loan_id": loan_id,
        "portal": portal,
        "status": "error",
        "message": (
            "FEMA flood map rendered corrupted/unreadable; no flood zone could be "
            "determined. No PDF was filed and no determination was written; "
            "Initial Processing Comments update attempted and loan surfaced as errored."
        ),
        "timestamp": datetime.now().isoformat(),
        "borrowers_processed": [borrower],
        "address": address,
        "expiration_check": ifd_expiration,
    }
    if extraction_log is not None:
        result["flood_zone_extraction"] = extraction_log
    return result


def _build_initial_processing_fields(user_name: str) -> list[dict[str, Any]]:
    """Build the field-write payload that presses the flood determination button.

    The button's timestamp is split across two fields (date and time) plus a
    user field. We split the same canonical stamp the other agents write as a
    single field, so the value is consistent across the platform.
    """
    # Client-local, not UTC: both halves are read on the Initial Processing screen.
    now = client_now()
    return [
        {"id": INITIAL_PROC_FLOOD_USER_FIELD_ID, "value": user_name},
        # CX.INITIAL.FLOOD.DETER.DATE is a date-typed Encompass field that
        # requires ISO 'yyyy-MM-dd' with no timezone offset — sending US
        # 'MM/dd/yyyy' fails with a Serialization 400. (Note: this differs from
        # the 2365 Determination Date field, which accepts a 'MM/dd/yyyy ...'
        # string.)
        {"id": INITIAL_PROC_FLOOD_DATE_FIELD_ID, "value": now.strftime("%Y-%m-%d")},
        {"id": INITIAL_PROC_FLOOD_TIME_FIELD_ID, "value": now.strftime("%I:%M:%S %p")},
    ]


def press_initial_processing_button(
    access_token: str,
    api_server: str,
    loan_guid: str,
    loan_id: str,
    user_name: str,
) -> None:
    """Press the "Initial Flood Determination" Initial Processing button.

    Writes the button's user/date/time completion fields via the v3
    fieldWriter endpoint, which Encompass treats as the button being clicked.
    This is the final step of the Job Aid flow (pp. 67-70) and is only invoked
    after the determination fields are written and the FLOODSEARCH map is
    uploaded — the needs-review and no-digital-data branches return earlier and
    intentionally never press the button ("What If? — do NOT use the initial
    processing button").
    """
    fields = _build_initial_processing_fields(user_name)
    logger.info(
        f"ifd_press_initial_processing_button loan_id={loan_id} "
        f"fields={[f['id'] for f in fields]}"
    )
    update_custom_fields(access_token, api_server, loan_guid, loan_id, fields)
    logger.info(f"Pressed IFD initial processing button for loan {loan_id}")


async def process_loan(loan_id: str, portal: str) -> dict[str, Any]:
    """Process a single loan for Initial Flood Determination.

    Args:
        loan_id: The Encompass loan ID to process.
        portal: Portal type (must be ``"IFD"``).

    Returns:
        Result dictionary including ``status`` (``"success"``, ``"skipped"``,
        ``"needs_review"``, or ``"error"``), the extracted flood zone payload,
        and the validation/expiration check metadata for the orchestrator UI.
    """
    logger.info(f"Processing loan {loan_id} for IFD portal")

    if portal.upper() != IFD_EFOLDER_PORTAL:
        raise ValueError(f"Invalid portal: {portal}. This agent only supports IFD portal.")

    secrets = get_cached_secrets()

    try:
        loan_id, loan_guid, api_server, access_token = connection(secrets, loan_id)

        eligibility_check = get_loan_status_and_whats_up(
            access_token, api_server, loan_guid, loan_id
        )

        eligibility_error = eligibility_check.get("error")
        if eligibility_error and eligibility_error.get("occurred"):
            error_type = eligibility_error.get("type", "unknown")
            error_message = eligibility_error.get("message", "Unknown error")
            error_details = eligibility_error.get("details", {})

            logger.error(
                f"Error during eligibility check for loan {loan_id}: {error_type} - {error_message}"
            )

            if error_type in ["api_error"] and error_details.get("status_code") in [401, 403, 404]:
                return {
                    "loan_id": loan_id,
                    "portal": portal,
                    "status": "error",
                    "message": f"Eligibility check failed: {error_message}",
                    "timestamp": datetime.now().isoformat(),
                    "borrowers_processed": [],
                    "error": {
                        "type": error_type,
                        "message": error_message,
                        "details": error_details,
                    },
                }
            else:
                logger.warning(
                    f"Eligibility check error for loan {loan_id}, proceeding (fail-open)"
                )

        if not eligibility_check.get("is_eligible"):
            exclusion_reason = eligibility_check.get("exclusion_reason", "Unknown reason")
            logger.warning(f"Loan {loan_id} is NOT eligible for processing: {exclusion_reason}")

            return {
                "loan_id": loan_id,
                "portal": portal,
                "status": "skipped",
                "message": f"Loan {loan_id} excluded from processing: {exclusion_reason}",
                "timestamp": datetime.now().isoformat(),
                "borrowers_processed": [],
                "exclusion_reason": exclusion_reason,
                "loan_status": eligibility_check.get("status"),
                "whats_up": eligibility_check.get("whats_up"),
            }

        logger.info(f"Loan {loan_id} passed eligibility check")

        logger.info(f"Checking IFD document expiration for loan {loan_id} (30-day threshold)")
        ifd_expiration = check_document_expiration(
            access_token, api_server, loan_guid, loan_id, IFD_BUCKET_NAME, days_threshold=30
        )

        if ifd_expiration.get("skip_processing"):
            days_old = ifd_expiration.get("days_old", "unknown")
            logger.info(
                f"Skipping IFD processing for loan {loan_id} - Document exists and is "
                f"only {days_old} days old"
            )

            return {
                "loan_id": loan_id,
                "portal": portal,
                "status": "skipped",
                "message": (
                    f"IFD document exists ({days_old} days old, not expired) - "
                    "skipping processing"
                ),
                "timestamp": datetime.now().isoformat(),
                "borrowers_processed": [],
                "skip_reason": "document_not_expired",
                "days_old": days_old,
                "document_date": ifd_expiration.get("document_date"),
                "expiration_check": ifd_expiration,
            }

        logger.info("IFD document expired or missing - proceeding with processing")

        # Read subject property address from the loan record
        address = get_property_address(access_token, api_server, loan_guid, loan_id)
        if not address.get("street") or not address.get("city") or not address.get("state"):
            raise ValueError(
                "Subject property address missing required street/city/state for NFHL search"
            )

        # Drive the NFHL viewer and capture FLOODSEARCH.pdf. A DETERMINISTIC
        # capture failure (page won't load, address unlocatable, blank/failed
        # render) is not worth retrying — it's a "no usable map" outcome: write
        # the IP comment and route to manual review, non-retryable. Only
        # genuinely transient failures (timeouts, browser/network hiccups)
        # re-raise to the retryable error handler so they can self-heal.
        logger.info(f"Capturing FEMA flood determination PDF for loan {loan_id}")
        try:
            pdf_content, pdf_size = await capture_fema_pdf_async(address)
        except LoanLockConflictError:
            raise
        except Exception as capture_error:
            if _is_transient_capture_error(capture_error):
                raise
            logger.warning(
                f"ifd_map_unavailable loan_id={loan_id} stage=capture "
                f"error_type={type(capture_error).__name__} error={capture_error}"
            )
            _write_no_data_ip_comment(access_token, api_server, loan_guid, loan_id)
            result = _no_usable_map_result(
                loan_id=loan_id,
                portal=portal,
                address=address,
                ifd_expiration=ifd_expiration,
                map_captured=False,
            )
            cleanup_result = cleanup_pdf_storage()
            if cleanup_result["files_deleted"] > 0:
                logger.info(f"Cleanup: {cleanup_result['files_deleted']} file(s) removed")
            return result

        # Best-effort: keep an S3 copy of the FEMA PDF for debugging. Do this
        # before vision extraction so low-confidence results still preserve the
        # artifact operators need for manual review.
        pdf_storage_dir = "/tmp/pdf_storage"
        os.makedirs(pdf_storage_dir, exist_ok=True)
        pdf_path = os.path.join(pdf_storage_dir, IFD_PDF_FILENAME)
        try:
            with open(pdf_path, "wb") as f:
                f.write(pdf_content)
            try:
                upload_into_s3(secrets, f"{loan_id}_FLOODSEARCH", pdf_path)
            except Exception as s3_error:
                logger.warning(f"S3 upload failed for FEMA PDF (continuing): {s3_error}")
        except Exception as fs_error:
            logger.warning(f"Could not persist FEMA PDF to local storage: {fs_error}")

        # Extract the flood zone FIRST — before the eFolder upload — so any
        # needs-review outcome can skip filing the PDF. Operations completes
        # those map captures manually, so only confident determinations proceed
        # to eFolder upload and Encompass field/button writes.
        logger.info(f"Extracting flood zone via Bedrock vision for loan {loan_id}")
        extraction: FloodZoneExtraction = extract_flood_zone(pdf_content)

        zone_found = extraction.zone_found.strip().lower()
        extraction_log = extraction.to_log_dict()

        # Corrupted/unreadable render per vision: the FEMA map painted but is
        # garbled (crosshatch mesh, scrambled tiles) so no zone can be read.
        # Write a dated Initial Processing Comments note, skip the PDF upload
        # and ALL determination / Initial-Processing writes, and surface the run
        # as a (retryable) error — a fresh capture may render cleanly next pass.
        # Checked before is_no_digital_data: a corrupted render is an error
        # outcome, whereas no-data is a terminal needs_review.
        if extraction.is_map_unreadable:
            # Client-local, not UTC: the M/D prefix is read by staff on the
            # Initial Processing Comments note.
            now = client_now()
            comment = f"{now.month}/{now.day} {MAP_UNREADABLE_COMMENT_BASE}"
            logger.warning(f"ifd_map_unreadable loan_id={loan_id} extraction={extraction_log}")
            _write_ip_comment(access_token, api_server, loan_guid, loan_id, comment)
            result = _map_unreadable_result(
                loan_id=loan_id,
                portal=portal,
                address=address,
                ifd_expiration=ifd_expiration,
                comment=comment,
                extraction_log=extraction_log,
            )
            cleanup_result = cleanup_pdf_storage()
            if cleanup_result["files_deleted"] > 0:
                logger.info(f"Cleanup: {cleanup_result['files_deleted']} file(s) removed")
            return result

        # No usable map per vision (FEMA "no digital data available", or a
        # blank/failed render the vision model couldn't classify). Same terminal outcome
        # as a deterministic capture failure: write the IP comment, skip the
        # PDF upload and determination writes, route to manual review.
        if extraction.is_no_digital_data:
            logger.warning(f"ifd_no_digital_data loan_id={loan_id} extraction={extraction_log}")
            _write_no_data_ip_comment(access_token, api_server, loan_guid, loan_id)
            result = _no_usable_map_result(
                loan_id=loan_id,
                portal=portal,
                address=address,
                ifd_expiration=ifd_expiration,
                map_captured=True,
                extraction_log=extraction_log,
            )
            cleanup_result = cleanup_pdf_storage()
            if cleanup_result["files_deleted"] > 0:
                logger.info(f"Cleanup: {cleanup_result['files_deleted']} file(s) removed")
            return result

        # Short-circuit when the vision call did not confidently classify.
        # An empty `flood_zone` paired with `zone_found="yes"` is intentional
        # — it means the property's polygon was unshaded and the property
        # is confidently NOT in any FEMA-mapped SFHA. That outcome writes the
        # 2366 checkbox as False and the 2367/541 dropdowns as "X" (minimal hazard).
        # Only `success=False` or `zone_found != "yes"` is unconfident. We do
        # not file the captured PDF for these; the run is surfaced as errored
        # (needs-review workflow deferred — BRAVO-135).
        if not extraction.success or zone_found != "yes":
            uncertain_reason = (
                "extraction_failed" if not extraction.success else f"zone_found={zone_found}"
            )
            logger.warning(
                f"ifd_map_uncertain_errored loan_id={loan_id} reason={uncertain_reason} "
                f"extraction={extraction_log}"
            )
            result = {
                "loan_id": loan_id,
                "portal": portal,
                # Surfaced as error (not needs_review): the needs-review workflow
                # is deferred (BRAVO-135), so an unconfident extraction is errored
                # for retry/visibility until that lands.
                "status": "error",
                "message": (
                    "Flood zone could not be confidently extracted from the FEMA PDF; "
                    "run surfaced as errored for retry/visibility."
                ),
                "timestamp": datetime.now().isoformat(),
                "borrowers_processed": [
                    {
                        "borrower_name": address.get("formatted")
                        or f"{address.get('street', '')}, "
                        f"{address.get('city', '')}, {address.get('state', '')}",
                        "state": address.get("state", ""),
                        "search_url": NFHL_VIEWER_URL,
                        "status": "Flood zone could not be classified",
                        "efolder_uploaded": False,
                        "encompass_fields_updated": False,
                        "map_captured": True,
                    }
                ],
                "address": address,
                "flood_zone_extraction": extraction_log,
                "expiration_check": ifd_expiration,
            }
            cleanup_result = cleanup_pdf_storage()
            if cleanup_result["files_deleted"] > 0:
                logger.info(f"Cleanup: {cleanup_result['files_deleted']} file(s) removed")
            return result

        # Upload the PDF to Encompass eFolder bucket 132 - Flood Search. Vision
        # extraction ran first so manual-review branches above could skip the
        # upload entirely; only confident determinations reach this point.
        try:
            logger.info(f"Uploading FEMA PDF to Encompass eFolder for loan {loan_id}")
            efolder_status = upload_file_into_efolder(
                secrets.get("ENCOMPASS_USERNAME"),
                api_server,
                access_token,
                loan_id,
                loan_guid,
                IFD_EFOLDER_PORTAL,
                IFD_PDF_FILENAME,
                pdf_content,
                pdf_size,
            )
            logger.info(f"FEMA PDF uploaded to Encompass eFolder: {efolder_status}")
            encompass_updated = True
        except LoanLockConflictError:
            raise
        except Exception as upload_error:
            logger.error(f"Failed to upload FEMA PDF to Encompass eFolder: {upload_error}")
            efolder_status = f"Upload failed: {str(upload_error)}"
            encompass_updated = False

        flood_zone_value = extraction.flood_zone.strip()
        # `written_zone` is what lands in 2367/541: the actual FEMA zone code
        # the vision model read (X, D, C, AE, …), defaulting to "X" only when the map is
        # uncolored / no code is visible (FEMA Zone X = minimal hazard). A named
        # non-SFHA zone like D or C is therefore preserved, not flattened to X.
        written_zone = flood_zone_value or "X"
        # 2366 (in flood zone) is derived from SFHA status, NOT from "is a code
        # present". Only A*/V* zones are SFHAs; X (incl. the orange/0.2% shaded
        # zone), B, C, D and uncolored are NOT in a flood zone → False.
        in_flood_zone = _is_sfha_zone(written_zone)
        # Client-local, not UTC: written to the 2365 Determination Date field.
        determination_date = encompass_timestamp()
        logger.info(
            f"ifd_zone_extracted loan_id={loan_id} flood_zone='{flood_zone_value}' "
            f"written_zone='{written_zone}' in_flood_zone={in_flood_zone} "
            f"sfha={extraction.sfha or 'unclear'}"
        )
        # Cross-check: the deterministic SFHA classification is authoritative,
        # but log when the vision model's `sfha` guess disagrees (observability
        # on extraction drift / prompt regressions).
        sfha_guess = (extraction.sfha or "").strip().lower()
        if sfha_guess in ("yes", "no") and (sfha_guess == "yes") != in_flood_zone:
            logger.warning(
                f"ifd_sfha_mismatch loan_id={loan_id} written_zone='{written_zone}' "
                f"classified_in_flood_zone={in_flood_zone} vision_sfha='{sfha_guess}'"
            )

        # Build the field-write payload. All fields are ALWAYS written so the
        # determination stays internally consistent:
        # - 2366 boolean checkbox — the "is this property in a flood zone?"
        #   signal (SFHA status; see `_is_sfha_zone`). The Python `bool`
        #   serializes to a JSON `true`/`false` literal, which Encompass requires
        #   for this field (string "Y"/"N" produces a Serialization error).
        # - 2367 and 541 zone dropdowns — the actual FEMA zone code, or "X" when
        #   the map is uncolored / unlabeled. Writing them unconditionally also
        #   overwrites any stale code from a prior run.
        fields_to_update: list[dict[str, Any]] = [
            {"id": IN_FLOOD_ZONE_CHECKBOX_FIELD_ID, "value": in_flood_zone},
            {"id": FLOOD_ZONE_DROPDOWN_FIELD_ID, "value": written_zone},
            {"id": OPERATIONS_FLOOD_ZONE_DROPDOWN_FIELD_ID, "value": written_zone},
            {"id": DETERMINATION_DATE_FIELD_ID, "value": determination_date},
        ]

        # Decision log right before we hand off to Encompass. Makes it
        # possible to grep `ifd_field_write_decision` for any loan and
        # immediately see what was about to be written and which branch
        # (in-zone vs not-in-zone) the code took. The companion success/
        # error log comes from the try/except below.
        logger.info(
            f"ifd_field_write_decision loan_id={loan_id} in_flood_zone={in_flood_zone} "
            f"written_zone='{written_zone}' "
            f"fields_to_write={[f['id'] for f in fields_to_update]}"
        )

        # Persist to Encompass first; if the field write fails we don't
        # pollute bucket 132 with a half-completed determination.
        field_ids = [f["id"] for f in fields_to_update]
        try:
            update_custom_fields(
                access_token,
                api_server,
                loan_guid,
                loan_id,
                fields_to_update,
            )
            logger.info(
                f"Encompass fields {field_ids} updated for loan {loan_id} "
                f"(in_flood_zone={in_flood_zone}, zone='{written_zone}')"
            )
            field_updated = True
            field_update_error: str | None = None
        except LoanLockConflictError:
            raise
        except Exception as field_error:
            logger.error(f"Failed to update Encompass fields {field_ids}: {field_error}")
            field_updated = False
            field_update_error = str(field_error)

        # Final step of the Job Aid: press the "Initial Flood Determination"
        # Initial Processing button. Only do so once the determination fields
        # and the FLOODSEARCH upload have both landed — a half-completed loan
        # must stay open for manual review rather than be marked done. A failed
        # button press keeps the run out of "success" (it drops to "partial")
        # so the loan is re-picked rather than silently left un-pressed.
        button_pressed = False
        button_press_error: str | None = None
        if field_updated and encompass_updated:
            try:
                press_initial_processing_button(
                    access_token,
                    api_server,
                    loan_guid,
                    loan_id,
                    secrets.get("ENCOMPASS_USERNAME") or "system",
                )
                button_pressed = True
            except LoanLockConflictError:
                raise
            except Exception as press_error:
                logger.error(
                    f"Failed to press IFD initial processing button for loan {loan_id}: "
                    f"{press_error}"
                )
                button_press_error = str(press_error)

        if field_updated and encompass_updated and button_pressed:
            overall_status = "success"
            _clear_ip_comment_if_ifd_authored(access_token, api_server, loan_guid, loan_id)
        elif field_updated or encompass_updated:
            overall_status = "partial"
        else:
            overall_status = "error"

        result = {
            "loan_id": loan_id,
            "portal": portal,
            "status": overall_status,
            "message": (
                f"Processed IFD for loan {loan_id}: flood zone {written_zone} "
                f"(SFHA={extraction.sfha or 'unclear'})"
            ),
            "timestamp": datetime.now().isoformat(),
            "borrowers_processed": [
                {
                    "borrower_name": address.get("formatted")
                    or f"{address.get('street', '')}, "
                    f"{address.get('city', '')}, {address.get('state', '')}",
                    "state": address.get("state", ""),
                    "search_url": NFHL_VIEWER_URL,
                    "pdf_filename": IFD_PDF_FILENAME,
                    "status": efolder_status if encompass_updated else str(efolder_status),
                    "efolder_uploaded": encompass_updated,
                    "encompass_fields_updated": field_updated and encompass_updated,
                    "in_flood_zone": in_flood_zone,
                    "flood_zone": written_zone,
                    "sfha": extraction.sfha,
                    "panel_number": extraction.panel_number,
                    "panel_effective_date": extraction.panel_effective_date,
                    "community_name": extraction.community_name,
                    "community_id": extraction.community_id,
                }
            ],
            "address": address,
            "flood_zone_extraction": extraction_log,
            "expiration_check": ifd_expiration,
            "field_update": {
                "field_ids": field_ids,
                "in_flood_zone": in_flood_zone,
                "flood_zone": written_zone,
                "determination_date": determination_date,
                "success": field_updated,
                "error": field_update_error,
            },
            "initial_processing_button": {
                "pressed": button_pressed,
                "error": button_press_error,
            },
        }

        if loan_id not in processed_loans_storage:
            processed_loans_storage[loan_id] = []
        processed_loans_storage[loan_id].append(result)

        cleanup_result = cleanup_pdf_storage()
        if cleanup_result["files_deleted"] > 0:
            logger.info(f"Cleanup: {cleanup_result['files_deleted']} file(s) removed")

        return result

    except LoanLockConflictError as e:
        error_msg = f"Loan is locked by '{e.locked_by}' — cannot proceed"
        logger.warning(error_msg)

        lock_result: dict[str, Any] = {
            "loan_id": loan_id,
            "portal": portal,
            "status": "error",
            "message": error_msg,
            "retryable": False,
            "timestamp": datetime.now().isoformat(),
            "borrowers_processed": [],
        }

        if loan_id not in processed_loans_storage:
            processed_loans_storage[loan_id] = []
        processed_loans_storage[loan_id].append(lock_result)

        cleanup_pdf_storage()

        return lock_result

    except Exception as e:
        logger.exception(f"Error processing loan {loan_id}: {e}")

        result = {
            "loan_id": loan_id,
            "portal": portal,
            "status": "error",
            "message": f"Error processing loan {loan_id}: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "borrowers_processed": [],
        }

        if loan_id not in processed_loans_storage:
            processed_loans_storage[loan_id] = []
        processed_loans_storage[loan_id].append(result)

        cleanup_pdf_storage()

        return result


def get_loan_property_address(loan_id: str) -> dict[str, Any]:
    """Look up the subject property address for a loan via Encompass."""
    try:
        secrets = get_cached_secrets()
        loan_id, loan_guid, api_server, access_token = connection(secrets, loan_id)
        address = get_property_address(access_token, api_server, loan_guid, loan_id)
        return {
            "loan_id": loan_id,
            "address": address,
        }
    except Exception as e:
        logger.exception(f"Error getting property address for loan {loan_id}: {e}")
        return {
            "loan_id": loan_id,
            "error": str(e),
            "address": {},
        }


async def get_loan_status_async(loan_id: str) -> dict[str, Any]:
    """Get status of a processed loan."""
    try:
        if loan_id not in processed_loans_storage or not processed_loans_storage[loan_id]:
            return {
                "success": False,
                "error": f"Loan {loan_id} not found. It may not have been processed yet.",
            }

        most_recent = processed_loans_storage[loan_id][-1]

        return {
            "success": True,
            "loan_id": most_recent["loan_id"],
            "status": most_recent["status"],
            "message": most_recent["message"],
            "timestamp": most_recent["timestamp"],
            "borrowers_processed": most_recent.get("borrowers_processed", []),
        }
    except Exception as e:
        logger.exception(f"Error getting loan status for {loan_id}: {e}")
        return {"success": False, "error": f"Error checking loan status: {str(e)}"}


async def get_all_loans_async() -> dict[str, Any]:
    """Get all processed loans."""
    try:
        all_loans = []
        for _loan_id, loan_list in processed_loans_storage.items():
            all_loans.extend(loan_list)

        return {"success": True, "loans": all_loans, "count": len(all_loans)}
    except Exception as e:
        logger.exception(f"Error getting all loans: {e}")
        return {"success": False, "error": f"Error fetching loans: {str(e)}"}


async def get_stats_async() -> dict[str, Any]:
    """Get processing statistics."""
    try:
        all_loans = []
        for _loan_id, loan_list in processed_loans_storage.items():
            all_loans.extend(loan_list)

        total_loans = len(all_loans)
        successful_loans = len([loan for loan in all_loans if loan.get("status") == "success"])
        failed_loans = len([loan for loan in all_loans if loan.get("status") == "error"])

        return {
            "success": True,
            "stats": {
                "total_loans": total_loans,
                "successful_loans": successful_loans,
                "failed_loans": failed_loans,
                "success_rate": (successful_loans / total_loans * 100) if total_loans > 0 else 0,
            },
        }
    except Exception as e:
        logger.exception(f"Error getting stats: {e}")
        return {"success": False, "error": f"Error fetching stats: {str(e)}"}
