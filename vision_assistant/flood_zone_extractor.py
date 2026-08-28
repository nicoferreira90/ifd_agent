"""Extract flood zone information from a FEMA NFHL viewer PDF capture.

Structurally mirrors `tools/src/af/tools/vision/seci_prior_note_address_extractor.py`:
PyMuPDF renders the captured PDF to PNG pages, Bedrock Converse invokes a Claude
vision model with a single forced `toolSpec`, and the parsed `input` is mapped to
a frozen dataclass.

The input PDFs come from the FEMA NFHL Web AppViewer capture flow in
`playwright_assistant/capturePDF_async.py` — not from the older msc.fema.gov
locator-map path, which produced blank canvases in headless Chromium.

This module is colocated with the IFD agent (rather than `tools/`) so we can
iterate on prompt, schema, and DPI choices without coordinating tools-package
releases. Promote to `tools/` once the contract stabilizes.
"""

import os
import re
from dataclasses import dataclass, fields
from typing import Any

import boto3
from af.tools import logger
from botocore.exceptions import BotoCoreError, ClientError

try:
    import fitz
except ImportError:  # pragma: no cover - exercised in runtimes without PyMuPDF installed
    fitz = None

DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
DEFAULT_MAX_PAGES = 2
DEFAULT_DPI = 150
# Bedrock Converse enforces its 5 MB per-image cap on the BASE64-ENCODED bytes
# (5,242,880), so the raw PNG must stay under 5,242,880 * 3/4 = 3,932,160.
# Observed live (2026-07-13, loan 47026071911): a 3,954,187-byte 150-DPI render
# encoded to 5,272,252 bytes and the whole run failed with ValidationException.
# Keep a cushion under the exact cap to absorb PNG-compression noise.
MAX_PAGE_PNG_BYTES = 3_900_000
_MIN_RENDER_DPI = 72
_MAX_DOWNSCALE_ATTEMPTS = 3
TOOL_NAME = "extract_flood_zone"
_CONTROLLED_ERROR_MESSAGES = {
    "empty_pdf_bytes",
    "zero_page_pdf",
    "pymupdf_not_installed",
    "oversized_page_render",
    "content_blocks_not_list",
    "tool_input_not_dict",
    "missing_tool_use",
}
_CLIENT_ERROR_CODE_MAP = {
    "AccessDeniedException": "access_denied",
    "ThrottlingException": "throttled",
    "TooManyRequestsException": "throttled",
    "ValidationException": "validation_error",
    "ModelTimeoutException": "model_timeout",
    "ModelErrorException": "model_error",
    "InternalServerException": "service_error",
    "ServiceUnavailableException": "service_unavailable",
}

# FEMA flood zone classification. SFHA = Special Flood Hazard Area, the
# high-risk designation that triggers federal mandatory-purchase flood
# insurance. A-prefixed and V-prefixed zones (including legacy numbered
# A1–A30 / V1–V30) are SFHA; B, C, D, X (incl. X500) are not.
_SFHA_ZONE_CODES: frozenset[str] = frozenset(
    {"A", "AE", "AH", "AO", "AR", "A99", "V", "VE"}
    | {f"A{n}" for n in range(1, 31)}
    | {f"V{n}" for n in range(1, 31)}
)
_NON_SFHA_ZONE_CODES: frozenset[str] = frozenset({"B", "C", "D", "X", "X500"})


def _normalize_flood_zone(raw: str) -> str:
    """Canonicalize a model-extracted FEMA zone code.

    Uppercases and strips a leading ``"ZONE "`` prefix the model
    occasionally emits (e.g. ``"Zone AE"`` → ``"AE"``, ``"ae"`` → ``"AE"``).
    Returns ``""`` for empty input. Applied at parse time so every downstream
    consumer (validation, logs, Encompass field writes) sees the canonical
    form.
    """
    normalized = raw.strip().upper()
    if normalized.startswith("ZONE "):
        normalized = normalized[len("ZONE ") :].strip()
    return normalized


def _expected_sfha_for_zone(zone: str) -> str:
    """Return the expected SFHA designation for a canonical FEMA zone code.

    Returns ``"yes"`` / ``"no"`` for recognized codes, and ``""`` for
    unrecognized codes — callers should treat ``""`` as "don't enforce" so
    rare-but-legitimate codes outside the known set are not over-rejected.
    """
    if zone in _SFHA_ZONE_CODES:
        return "yes"
    if zone in _NON_SFHA_ZONE_CODES:
        return "no"
    return ""


# Schema is tailored for FEMA NFHL viewer captures. `reasoning` MUST stay the
# first property (dict insertion order is preserved into the serialized
# schema): the model fills fields in schema order, so it deliberates about the
# marker location and shading before committing to a classification. Eval'd
# 2026-07 on ops-flagged misclassified captures: prompt + reasoning field
# scored 58/60 across a 10-run stability sweep vs 2/4 for the same prompt
# without the field (see ~/Desktop/moder/flood-eval). `zone_found` is the
# explicit gate the IFD pipeline uses to decide whether the map was classified
# confidently enough to write the determination fields (2366 and 2367 are both
# written when "yes" — 2367 gets the zone code when in a zone, or an empty
# string to clear it for the unshaded/no-zone case, which legitimately yields
# flood_zone="" with zone_found="yes"). `no_digital_data` is a separate
# terminal signal: when "yes" the pipeline skips the PDF upload and
# determination writes entirely and routes to manual review.
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": (
                "FIRST, before deciding anything else: describe what you observe on "
                "the map — where the property pin / searched location sits, what "
                "shading color (if any) covers or borders it, any zone text labels "
                "visible near it, panel/legend information, and any rendering "
                "problems. Then state which classification rule (0a corrupted render, "
                "0 no digital data, 1 explicit text label, 2 color default) you "
                "applied and why. Mention anything ambiguous that made the call "
                "difficult."
            ),
        },
        "flood_zone": {
            "type": "string",
            "description": (
                "FEMA flood zone code (e.g., X, AE, A, VE, AH, AO, D). "
                "If an explicit zone label is shown on or near the property polygon, "
                "use that. Otherwise classify by shading color per the user prompt: "
                "teal/light-blue → AE; orange/peach → X; unshaded → empty string "
                "(the property is not in any FEMA-mapped flood zone)."
            ),
        },
        "sfha": {
            "type": "string",
            "description": (
                "Whether the property is in a Special Flood Hazard Area. Use yes, no, or unclear."
            ),
        },
        "panel_number": {
            "type": "string",
            "description": "FEMA FIRM panel number shown on the map.",
        },
        "panel_effective_date": {
            "type": "string",
            "description": "FEMA FIRM panel effective date as shown on the map.",
        },
        "community_name": {
            "type": "string",
            "description": "FEMA community name shown on the map.",
        },
        "community_id": {
            "type": "string",
            "description": "FEMA community ID shown on the map.",
        },
        "zone_found": {
            "type": "string",
            "description": (
                "Whether a flood zone was confidently extracted from this map. "
                "Use yes, no, or unclear."
            ),
        },
        "no_digital_data": {
            "type": "string",
            "description": (
                "Whether the map has NO FEMA digital flood data for the property at "
                "all — e.g. it shows 'NO DIGITAL DATA AVAILABLE', the area is marked "
                "as unmapped/not included in the NFHL, or no flood layer rendered for "
                "the location. Use yes or no. This is DIFFERENT from an unshaded "
                "property, which has data but simply no flood-zone overlay (that case "
                "is no_digital_data='no')."
            ),
        },
        "map_unreadable": {
            "type": "string",
            "description": (
                "Whether the map rendered but is visually CORRUPTED / garbled / not "
                "properly displayed so NO flood zone can be determined — e.g. the "
                "view is covered by a broken crosshatch or diagonal-mesh pattern, "
                "scrambled or torn tiles, heavy static/noise, or the flood overlay "
                "smeared across the whole image. Use yes or no. This is DIFFERENT "
                "from no_digital_data (FEMA simply has no data for the area) and from "
                "an unshaded property (a clean, readable map with no overlay). Only "
                "set 'yes' when the image itself is too corrupted to read."
            ),
        },
    },
    "required": [
        "reasoning",
        "flood_zone",
        "sfha",
        "panel_number",
        "panel_effective_date",
        "community_name",
        "community_id",
        "zone_found",
        "no_digital_data",
        "map_unreadable",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class FloodZoneExtraction:
    """Structured flood zone information extracted from FEMA map images."""

    success: bool
    flood_zone: str = ""
    sfha: str = ""
    panel_number: str = ""
    panel_effective_date: str = ""
    community_name: str = ""
    community_id: str = ""
    zone_found: str = ""
    no_digital_data: str = ""
    map_unreadable: str = ""
    # The model's own account of what it saw and which rule it applied,
    # emitted before the classification fields (schema order). Not used for
    # any decision — kept for CloudWatch debuggability of misclassifications.
    reasoning: str = ""
    error: str | None = None
    model_id: str | None = None
    pages_rendered: int = 0
    dpi: int = DEFAULT_DPI

    @property
    def is_no_digital_data(self) -> bool:
        """Return whether the map had no FEMA digital flood data for the property.

        This is a terminal-but-non-error outcome distinct from a low-confidence
        extraction: the website itself had nothing to classify (the FEMA
        "NO DIGITAL DATA AVAILABLE" state). The IFD pipeline routes these to
        manual review without uploading a PDF or pressing Initial Processing.
        """
        return self.no_digital_data.strip().lower() == "yes"

    @property
    def is_map_unreadable(self) -> bool:
        """Return whether the captured map rendered but is too corrupted to read.

        A terminal error outcome, distinct from ``is_no_digital_data`` (FEMA has
        no data for the area): the flood map painted as a garbled crosshatch /
        mesh or otherwise unreadable image, so no determination can be made. The
        IFD pipeline writes an Initial Processing Comments note, skips the eFolder
        upload and all determination / Initial-Processing writes, and surfaces
        the run as a (retryable) error so a fresh capture can be re-attempted.
        """
        return self.map_unreadable.strip().lower() == "yes"

    def has_primary_fields(self) -> bool:
        """Return whether the extraction is a coherent, actionable classification.

        Valid shapes when ``zone_found`` is ``"yes"``:

        - **In a flood zone**: ``flood_zone`` code is set AND ``sfha`` is
          one of ``"yes"`` / ``"no"`` AND the two agree with FEMA's
          deterministic mapping (e.g. ``"AE"``→``"yes"``, ``"X"``→``"no"``).
        - **Not in any flood zone**: ``flood_zone`` is empty AND ``sfha="no"``
          (the unshaded case — property polygon has no overlay).

        Rejection cases (route to needs_review):

        - Empty ``flood_zone`` with ``sfha`` anything other than ``"no"``.
        - Zone code present with empty / ``"unclear"`` / other-string ``sfha``.
        - Zone code present with a ``sfha`` that *contradicts* the zone's
          known FEMA classification (e.g. ``"AE"`` with ``sfha="no"`` —
          internally inconsistent, model is confused about either field).
        - Unrecognized zone codes are accepted as long as ``sfha`` is yes/no,
          to avoid over-rejecting rare-but-legitimate codes.
        - ``zone_found`` other than ``"yes"`` is rejected unconditionally —
          the method's contract is "is this confident enough to act on?", and
          ``"no"`` / ``"unclear"`` mean the model itself said it isn't.
        """
        # Self-contained contract check — don't rely on callers to filter on
        # zone_found first. The production gate in `extract_flood_zone` does
        # check it, but enforcing it here keeps the method robust if the call
        # site ever changes.
        if self.zone_found.strip().lower() != "yes":
            return False
        zone = _normalize_flood_zone(self.flood_zone)
        sfha = self.sfha.strip().lower()
        # Confident "not in any flood zone" outcome (unshaded property).
        if not zone:
            return sfha == "no"
        # Zone code present: SFHA must be a definite yes/no — `"unclear"` or
        # any other string means the model wasn't fully confident even though
        # it picked a zone, so route to human review.
        if sfha not in {"yes", "no"}:
            return False
        # If we recognize the zone code, enforce FEMA's deterministic
        # zone→SFHA mapping. Unknown codes pass through (empty expected).
        expected = _expected_sfha_for_zone(zone)
        if expected and expected != sfha:
            return False
        return True

    def to_log_dict(self) -> dict[str, Any]:
        """Return structured logging metadata (safe to emit)."""
        return {
            "success": self.success,
            "flood_zone": self.flood_zone,
            "sfha": self.sfha,
            "panel_number": self.panel_number,
            "panel_effective_date": self.panel_effective_date,
            "community_name": self.community_name,
            "community_id": self.community_id,
            "zone_found": self.zone_found,
            "no_digital_data": self.no_digital_data,
            "map_unreadable": self.map_unreadable,
            # Bounded so a long deliberation can't bloat log lines / results.
            "reasoning": self.reasoning[:2000],
            "error": self.error,
            "model_id": self.model_id,
            "pages_rendered": self.pages_rendered,
            "dpi": self.dpi,
        }


def extract_flood_zone(
    pdf_bytes: bytes,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    dpi: int = DEFAULT_DPI,
    model_id: str | None = None,
    bedrock_client: Any | None = None,
) -> FloodZoneExtraction:
    """Extract flood zone information from a rasterized FEMA NFHL viewer PDF.

    Returns `success=False` for any rendering, Bedrock, or response-shape failure.
    Callers decide whether those failures block downstream workflows.

    Args:
        pdf_bytes: Raw PDF bytes captured from the FEMA NFHL viewer.
        max_pages: Number of pages from the PDF to send to the model.
        dpi: PNG render DPI. Higher = larger payload + better legibility.
        model_id: Override the Bedrock model ID (defaults to DEFAULT_MODEL_ID).
        bedrock_client: Optional pre-built `bedrock-runtime` client (used in tests).

    Returns:
        `FloodZoneExtraction` describing what the model saw on the map.
    """
    selected_model_id = model_id or os.getenv("FLOOD_ZONE_EXTRACTOR_MODEL_ID") or DEFAULT_MODEL_ID

    try:
        png_pages = _render_pdf_pages_to_png(pdf_bytes, max_pages=max_pages, dpi=dpi)
    except Exception as exc:
        error_code = _stable_error("render_failed", exc)
        logger.warning(
            "ifd_flood_zone_extraction_failed "
            f"phase=render error_type={type(exc).__name__} error_code={error_code}"
        )
        return FloodZoneExtraction(
            success=False,
            error=error_code,
            model_id=selected_model_id,
            dpi=dpi,
        )

    if not png_pages:
        logger.warning("ifd_flood_zone_extraction_failed phase=render error=no_pages")
        return FloodZoneExtraction(
            success=False,
            error="no_pages_rendered",
            model_id=selected_model_id,
            dpi=dpi,
        )

    total_png_bytes = sum(len(page) for page in png_pages)
    max_png_bytes = max(len(page) for page in png_pages)
    logger.info(
        "ifd_flood_zone_render_complete "
        f"pages_rendered={len(png_pages)} dpi={dpi} "
        f"total_png_bytes={total_png_bytes} max_page_bytes={max_png_bytes}"
    )

    try:
        client = bedrock_client or boto3.client("bedrock-runtime")
        logger.info(
            "ifd_flood_zone_bedrock_request_start "
            f"model_id={selected_model_id} pages={len(png_pages)} tool_name={TOOL_NAME}"
        )
        response = client.converse(
            modelId=selected_model_id,
            messages=[_build_user_message(png_pages)],
            toolConfig={
                "tools": [
                    {
                        "toolSpec": {
                            "name": TOOL_NAME,
                            "description": (
                                "Extract FEMA flood zone information from a FEMA National "
                                "Flood Hazard Layer (NFHL) viewer map capture."
                            ),
                            "inputSchema": {"json": SCHEMA},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": TOOL_NAME}},
            },
            # 2048, not 512: the leading `reasoning` field can run long, and the
            # gate fields (`zone_found`, `no_digital_data`, `map_unreadable`)
            # serialize LAST — a truncated response silently blanks them and
            # turns a correct classification into an unconfident-errored run.
            inferenceConfig={"maxTokens": 2048, "temperature": 0},
        )
        logger.info(f"ifd_flood_zone_bedrock_response_received model_id={selected_model_id}")
    except Exception as exc:
        error_code = _stable_error("bedrock_client_error", exc)
        detail = " ".join(str(exc).split())[:500]
        logger.warning(
            "ifd_flood_zone_extraction_failed "
            f"phase=bedrock error_type={type(exc).__name__} error_code={error_code} "
            f"detail={detail!r}"
        )
        return FloodZoneExtraction(
            success=False,
            error=error_code,
            model_id=selected_model_id,
            pages_rendered=len(png_pages),
            dpi=dpi,
        )

    try:
        tool_input = _extract_tool_input(response)
        extraction = _flood_zone_extraction_from_tool_input(
            tool_input,
            model_id=selected_model_id,
            pages_rendered=len(png_pages),
            dpi=dpi,
        )
    except Exception as exc:
        error_code = _stable_error("malformed_tool_use_response", exc)
        logger.warning(
            "ifd_flood_zone_extraction_failed "
            f"phase=response_parse error_type={type(exc).__name__} error_code={error_code}"
        )
        try:
            raw_message = response.get("output", {}).get("message", {})
            stop_reason = response.get("stopReason")
            content_blocks = raw_message.get("content", []) if isinstance(raw_message, dict) else []
            block_types: list[str] = []
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if isinstance(block, dict):
                        block_types.append(next(iter(block.keys()), "unknown"))
            raw_repr = repr(raw_message)
            if len(raw_repr) > 2000:
                raw_repr = raw_repr[:2000] + "...[truncated]"
            logger.warning(
                "ifd_flood_zone_extraction_response_dump "
                f"stop_reason={stop_reason} "
                f"content_block_count={len(content_blocks) if isinstance(content_blocks, list) else 0} "
                f"block_types={block_types} "
                f"raw_message={raw_repr}"
            )
        except Exception as log_exc:  # pragma: no cover - logging must not crash the path
            logger.warning(f"ifd_flood_zone_extraction_response_dump_failed error={log_exc}")
        return FloodZoneExtraction(
            success=False,
            error=error_code,
            model_id=selected_model_id,
            pages_rendered=len(png_pages),
            dpi=dpi,
        )

    if _zone_found_value(extraction) == "yes" and not extraction.has_primary_fields():
        logger.warning(
            "ifd_flood_zone_extraction_failed "
            f"phase=validation error=invalid_primary_fields extraction={extraction.to_log_dict()}"
        )
        return FloodZoneExtraction(
            success=False,
            error="invalid_primary_fields",
            model_id=selected_model_id,
            pages_rendered=len(png_pages),
            dpi=dpi,
            flood_zone=extraction.flood_zone,
            sfha=extraction.sfha,
            panel_number=extraction.panel_number,
            panel_effective_date=extraction.panel_effective_date,
            community_name=extraction.community_name,
            community_id=extraction.community_id,
            zone_found=extraction.zone_found,
            reasoning=extraction.reasoning,
        )

    logger.info(f"ifd_flood_zone_extraction_success extraction={extraction.to_log_dict()}")
    return extraction


def _render_pdf_pages_to_png(
    pdf_bytes: bytes, *, max_pages: int = DEFAULT_MAX_PAGES, dpi: int = DEFAULT_DPI
) -> list[bytes]:
    if not pdf_bytes:
        raise ValueError("empty_pdf_bytes")
    if fitz is None:
        raise RuntimeError("pymupdf_not_installed")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_count = len(doc)
        if page_count == 0:
            raise ValueError("zero_page_pdf")

        pages_to_render = min(page_count, max(1, max_pages))
        png_pages: list[bytes] = []
        for page_index in range(pages_to_render):
            png_pages.append(
                _render_page_png_within_budget(doc[page_index], page_index=page_index, dpi=dpi)
            )
        return png_pages
    finally:
        doc.close()


def _render_page_png_within_budget(page: Any, *, page_index: int, dpi: int) -> bytes:
    """Render one page to PNG, downscaling DPI until it fits the Bedrock cap.

    Pages that fit at the requested DPI are returned untouched (the common
    case). An oversized page is re-rendered at a proportionally reduced DPI —
    PNG size tracks pixel count (~dpi^2), so one attempt normally lands under
    budget; we aim 5% below it to absorb compression-ratio noise. If the page
    still exceeds the cap at `_MIN_RENDER_DPI`, raise the controlled
    ``oversized_page_render`` error instead of sending a request Bedrock is
    guaranteed to reject.
    """
    png_bytes: bytes = page.get_pixmap(dpi=dpi, alpha=False).tobytes("png")
    current_dpi = dpi
    for _ in range(_MAX_DOWNSCALE_ATTEMPTS):
        if len(png_bytes) <= MAX_PAGE_PNG_BYTES:
            return png_bytes
        scale = (MAX_PAGE_PNG_BYTES * 0.95 / len(png_bytes)) ** 0.5
        reduced_dpi = max(_MIN_RENDER_DPI, int(current_dpi * scale))
        if reduced_dpi >= current_dpi:
            reduced_dpi = current_dpi - 1
        if reduced_dpi < _MIN_RENDER_DPI:
            break
        candidate: bytes = page.get_pixmap(dpi=reduced_dpi, alpha=False).tobytes("png")
        logger.info(
            "ifd_flood_zone_page_downscaled "
            f"page={page_index} from_dpi={current_dpi} to_dpi={reduced_dpi} "
            f"from_bytes={len(png_bytes)} to_bytes={len(candidate)}"
        )
        png_bytes = candidate
        current_dpi = reduced_dpi
    if len(png_bytes) > MAX_PAGE_PNG_BYTES:
        raise ValueError("oversized_page_render")
    return png_bytes


# Prompt v3 (2026-07): locate-the-marker-first + authoritative color scheme.
# Byte-identical to the config eval'd at 58/60 over a 10-run stability sweep on
# ops-flagged captures; the residual miss is an AE->VE zone-code flip (2/10) on
# dense urban maps where adjacent SFHA polygons adjoin the marker — SFHA status
# (field 2366) is unaffected by that flip. Keep in sync with the local eval
# harness prompt (~/Desktop/moder/flood-eval/prompts/) when iterating.
_VISION_PROMPT = """\
These images are pages captured from FEMA's National Flood Hazard Layer (NFHL) \
viewer for a single subject property. Extract the FEMA flood zone information \
for the property's EXACT location (flood zone code, SFHA status, FIRM panel \
number, panel effective date, community name, community ID) via the required \
tool.

FINDING THE PROPERTY — do this FIRST, before applying any classification rule:

The property's exact location is the small cyan/teal square marker at the \
anchor point of the "Search result" popup (the popup shows the property's \
address; the marker sits immediately left of the popup body). That single \
point is what you must classify. Everything else on the map — including large \
or prominently-labeled flood zone polygons elsewhere in the view — is context, \
NOT the property.

CRITICAL: classify ONLY the shading directly under the marker. A flood zone \
that covers much of the map but does NOT cover the marker is irrelevant. A \
zone text label sitting on a river, lake, stream corridor, or any polygon that \
does not cover the marker must be IGNORED, no matter how prominent it is. Do \
NOT assume the map is centered on the property, and do NOT assume the largest \
or most clearly labeled zone is the property's zone.

CLASSIFICATION RULES — apply in this order:

0a. CORRUPTED RENDER (check FIRST). If the captured map image is visually \
corrupted or did not render properly — the view is covered by a broken \
crosshatch / diagonal-mesh pattern, scrambled or torn tiles, heavy static or \
noise, or the flood overlay is smeared across the whole image — so that you \
cannot reliably read the flood zone at the marker, set map_unreadable="yes", \
zone_found="no", no_digital_data="no", and stop. This is a RENDERING failure, \
NOT a no-data condition. In every other case (the map renders cleanly enough \
to classify, including unshaded and no-data maps) set map_unreadable="no".

0. NO DIGITAL DATA. If the map has no FEMA flood data for the property at all \
— it displays "NO DIGITAL DATA AVAILABLE", the area is marked as unmapped / \
not included in the NFHL, or no flood layer rendered over the location — set \
no_digital_data="yes" and zone_found="no", and stop. CRITICAL: this is NOT the \
same as an unshaded property. An unshaded property sits on a normal, fully \
rendered map with roads/parcels/aerial imagery visible and simply has no \
flood-zone color over it — that HAS data (no_digital_data="no", see rule 2). \
Only set no_digital_data="yes" when the flood map itself is absent/unavailable \
for the area. In all other cases set no_digital_data="no".

1. EXPLICIT TEXT ON THE MARKER'S POLYGON WINS. If a shaded polygon covers the \
marker AND that same polygon carries a zone code as text (e.g. "Zone AE", \
"Zone X", "Zone VE", "AREA WITH REDUCED FLOOD RISK DUE TO LEVEE Zone X"), use \
that exact code. Set zone_found="yes". The label must belong to the polygon \
that covers the marker — a label on a DIFFERENT polygon, however close or \
prominent, does not count and must not influence the classification.

2. COLOR-CODING DEFAULTS — when the polygon covering the marker has no text \
label, classify by its shading color. This color scheme is AUTHORITATIVE for \
these captures: never reinterpret the colors using outside knowledge of FEMA \
cartography. Orange/peach NEVER means Zone AE in this viewer. A zone label \
sitting on a differently-shaded polygon (e.g. "Zone AE" on a teal water body \
next to an orange band) does NOT extend to the polygon covering the marker — \
if the marker's polygon is orange/peach, the answer is Zone X even when a \
neighboring teal polygon is labeled AE.
   - TEAL / LIGHT-BLUE shading covering the marker = Special Flood Hazard \
Area. Default to **Zone AE**. flood_zone="AE", sfha="yes", zone_found="yes".
   - ORANGE / PEACH shading covering the marker = Other Area of Flood Hazard \
(0.2% annual chance, or shallow/limited-drainage flooding). Default to \
**Zone X**. flood_zone="X", sfha="no", zone_found="yes".
   - UNSHADED / WHITE — no flood-zone overlay covers the marker itself. The \
property is **NOT in any FEMA-mapped flood zone**, even if shaded zones exist \
elsewhere on the map. flood_zone="" (empty), sfha="no", zone_found="yes". Do \
NOT label this as Zone X — orange shading is what Zone X looks like; unshaded \
means there is no zone here.

3. SFHA STATUS: "yes" for zones AE / AO / AH / VE / AR / A / A99 (any zone \
with a Base Flood Elevation or depth requirement). "no" for zones X and D, \
and for unshaded (no flood zone).

4. zone_found="unclear" ONLY when: the marker sits exactly on the boundary \
between two differently-shaded zones; the marker or its immediate surroundings \
are hidden by the Search-result popup or another UI element so the shading at \
the marker cannot be determined; or the shading at the marker genuinely \
doesn't match the scheme above. Do NOT set "unclear" just because there's no \
text label — the shading (or absence of shading) at the marker alone is \
enough.

If panel number, effective date, community name, or community ID is not \
visible on the captured page, return an empty string for that field. That \
alone does not make the zone unclear.

Always fill zone_found ("yes"/"no"/"unclear"), no_digital_data ("yes"/"no"), \
and map_unreadable ("yes"/"no") explicitly — never leave them empty, even when \
the classification was difficult.\
"""


def _build_user_message(png_pages: list[bytes]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"text": _VISION_PROMPT}]
    for png_bytes in png_pages:
        content.append({"image": {"format": "png", "source": {"bytes": png_bytes}}})
    return {"role": "user", "content": content}


def _extract_tool_input(response: dict[str, Any]) -> dict[str, Any]:
    content_blocks = response.get("output", {}).get("message", {}).get("content", [])
    if not isinstance(content_blocks, list):
        raise ValueError("content_blocks_not_list")

    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        tool_use = block.get("toolUse")
        if not isinstance(tool_use, dict):
            continue
        if tool_use.get("name") != TOOL_NAME:
            continue
        tool_input = tool_use.get("input")
        if not isinstance(tool_input, dict):
            raise ValueError("tool_input_not_dict")
        return tool_input

    raise ValueError("missing_tool_use")


def _flood_zone_extraction_from_tool_input(
    tool_input: dict[str, Any], *, model_id: str, pages_rendered: int, dpi: int
) -> FloodZoneExtraction:
    values: dict[str, Any] = {}
    dataclass_field_names = {field.name for field in fields(FloodZoneExtraction)}
    metadata_fields = {"success", "error", "model_id", "pages_rendered", "dpi"}
    for field_name in sorted(dataclass_field_names - metadata_fields):
        values[field_name] = str(tool_input.get(field_name) or "").strip()
    # Canonicalize flood_zone once at the parse boundary so all downstream
    # consumers — validation (`has_primary_fields`), logs, the Encompass
    # field writer for 2367 — see uppercase codes with no "Zone "
    # prefix. Prevents "ae" or "Zone AE" from slipping into Encompass
    # writes where the dropdown expects strict canonical codes.
    values["flood_zone"] = _normalize_flood_zone(values["flood_zone"])
    return FloodZoneExtraction(
        success=True,
        model_id=model_id,
        pages_rendered=pages_rendered,
        dpi=dpi,
        **values,
    )


def _zone_found_value(extraction: FloodZoneExtraction) -> str:
    return extraction.zone_found.strip().lower()


def _stable_error(prefix: str, exc: Exception) -> str:
    if isinstance(exc, ClientError):
        error_code = str(exc.response.get("Error", {}).get("Code") or "")
        reason = _CLIENT_ERROR_CODE_MAP.get(error_code, "client_error")
        return f"{prefix}:{reason}"

    if isinstance(exc, BotoCoreError):
        return f"{prefix}:boto_core_error"

    message = str(exc).strip()
    if message in _CONTROLLED_ERROR_MESSAGES:
        return f"{prefix}:{message}"

    reason = _exception_type_reason(exc)
    if reason:
        return f"{prefix}:{reason}"
    return prefix


def _exception_type_reason(exc: Exception) -> str:
    name = type(exc).__name__
    if not name or name == "Exception":
        return "unexpected_error"
    reason = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return reason or "unexpected_error"
