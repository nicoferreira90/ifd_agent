"""Tests for IFD process_loan orchestration branches."""

from typing import Any
from unittest.mock import Mock

import pytest

from ifd_agent.encompass_functions import (
    _is_ifd_authored_ip_comment,
    _is_sfha_zone,
    process_loan,
)
from ifd_agent.encompass_mcp_tool import (
    STEP_CAPTURE_PDF_FEMA,
    STEP_EXTRACT_FLOOD_ZONE,
    STEP_UPLOAD_EFOLDER_IFD,
    EncompassMCPTool,
)
from ifd_agent.vision_assistant.flood_zone_extractor import FloodZoneExtraction


class _FakeTracker:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str | None]] = []

    def start_step(self, code: str) -> None:
        self.events.append(("start", code, None))

    def complete_step(
        self,
        code: str,
        state: str = "completed",
        notes: str | None = None,
    ) -> None:
        # Mirror real tracker: states other than "completed" produce a
        # distinct event tag so tests can assert on routing (needs_review,
        # errored, etc.) without inventing a separate method.
        event_tag = "complete" if state == "completed" else state
        self.events.append((event_tag, code, notes))

    def error_step(self, code: str, notes: str | None = None) -> None:
        self.events.append(("error", code, notes))


def _patch_common_process_loan_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.get_cached_secrets",
        lambda: {"ENCOMPASS_USERNAME": "ifd-user"},
    )
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.connection",
        lambda _secrets, loan_id: (loan_id, "loan-guid", "https://encompass.example", "token"),
    )
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.get_loan_status_and_whats_up",
        lambda *_args, **_kwargs: {"is_eligible": True},
    )
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.check_document_expiration",
        lambda *_args, **_kwargs: {"skip_processing": False, "has_document": False},
    )
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.get_property_address",
        lambda *_args, **_kwargs: {
            "street": "6123 Raleigh St",
            "city": "Spring Hill",
            "state": "FL",
            "postal_code": "34606",
            "formatted": "6123 Raleigh St, Spring Hill, FL 34606",
        },
    )
    monkeypatch.setattr("ifd_agent.encompass_functions.upload_into_s3", lambda *_args: None)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.cleanup_pdf_storage",
        lambda: {"files_deleted": 1, "total_size_freed": 123, "errors": [], "success": True},
    )
    # Default: no stale Initial Processing Comment on the loan, so the
    # success-path clear is a no-op unless a test overrides this to exercise
    # the clear/don't-clear guard directly.
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.get_custom_field_value",
        lambda *_args, **_kwargs: "",
    )


def _assert_initial_processing_button_pressed(
    update_custom_fields: Mock, result: dict[str, Any]
) -> None:
    """Assert the second field write pressed the Initial Processing button.

    The button stamp is split across user/date/time fields (unlike the single
    combined field the other agents write). All three must be populated, and
    the username comes from ENCOMPASS_USERNAME ("ifd-user" in the test fixture).
    """
    button_fields = update_custom_fields.call_args_list[1][0][4]
    by_id = {f["id"]: f["value"] for f in button_fields}
    assert set(by_id) == {
        "CX.INITIAL.FLOOD.DETER.USER",
        "CX.INITIAL.FLOOD.DETER.DATE",
        "CX.INITIAL.FLOOD.DETER.TIME",
    }
    assert by_id["CX.INITIAL.FLOOD.DETER.USER"] == "ifd-user"
    assert by_id["CX.INITIAL.FLOOD.DETER.DATE"]
    assert by_id["CX.INITIAL.FLOOD.DETER.TIME"]
    assert result["initial_processing_button"]["pressed"] is True
    assert result["initial_processing_button"]["error"] is None


@pytest.mark.asyncio
async def test_process_loan_skips_pdf_upload_on_low_confidence_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Low-confidence map is surfaced as errored (needs-review deferred,
    # BRAVO-135); the automation must still not file the captured PDF to eFolder.
    _patch_common_process_loan_dependencies(monkeypatch)
    events: list[str] = []

    async def fake_capture(_address: dict[str, str]) -> tuple[bytes, int]:
        return b"%PDF-1.7\n", 9

    def fake_extract(_pdf_content: bytes) -> FloodZoneExtraction:
        events.append("extract")
        return FloodZoneExtraction(
            success=True,
            sfha="unclear",
            panel_number="12053C0303E",
            panel_effective_date="1/15/2021",
            community_name="HERNANDO COUNTY UNINCORPORATED AREAS",
            zone_found="no",
        )

    upload_file_into_efolder = Mock(return_value="should-not-be-called")
    update_custom_fields = Mock()
    monkeypatch.setattr("ifd_agent.encompass_functions.capture_fema_pdf_async", fake_capture)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.upload_file_into_efolder", upload_file_into_efolder
    )
    monkeypatch.setattr("ifd_agent.encompass_functions.extract_flood_zone", fake_extract)
    monkeypatch.setattr("ifd_agent.encompass_functions.update_custom_fields", update_custom_fields)

    result = await process_loan("12345", "IFD")

    assert events == ["extract"]
    assert result["status"] == "error"
    upload_file_into_efolder.assert_not_called()
    borrower = result["borrowers_processed"][0]
    assert borrower["efolder_uploaded"] is False
    assert borrower["map_captured"] is True
    assert "pdf_filename" not in borrower
    update_custom_fields.assert_not_called()


@pytest.mark.asyncio
async def test_process_loan_writes_checkbox_and_dropdowns_when_in_flood_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Teal-shaded property: Haiku returns flood_zone="AE", sfha="yes",
    # zone_found="yes". We should write the determination fields:
    # - 2366 (in-flood-zone boolean checkbox; covers Zone X too, not strictly SFHA) = True
    # - 2367 (legacy FEMA Flood Zone dropdown) = "AE"
    # - 541 (Operations FEMA Flood Zone dropdown) = "AE"
    # The 2366 value must be a Python bool — Encompass serializes it to JSON
    # `true`/`false`, which the field's underlying boolean type requires.
    _patch_common_process_loan_dependencies(monkeypatch)

    async def fake_capture(_address: dict[str, str]) -> tuple[bytes, int]:
        return b"%PDF-1.7\n", 9

    def fake_extract(_pdf_content: bytes) -> FloodZoneExtraction:
        return FloodZoneExtraction(success=True, flood_zone="AE", sfha="yes", zone_found="yes")

    update_custom_fields = Mock()
    monkeypatch.setattr("ifd_agent.encompass_functions.capture_fema_pdf_async", fake_capture)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.upload_file_into_efolder",
        lambda *_a, **_k: "ok",
    )
    monkeypatch.setattr("ifd_agent.encompass_functions.extract_flood_zone", fake_extract)
    monkeypatch.setattr("ifd_agent.encompass_functions.update_custom_fields", update_custom_fields)

    result = await process_loan("12345", "IFD")

    assert result["status"] == "success"
    # Two field writes on the success path: the determination fields, then the
    # Initial Processing button press (the final Job Aid step).
    assert update_custom_fields.call_count == 2
    written_fields = update_custom_fields.call_args_list[0][0][4]
    assert written_fields[:3] == [
        {"id": "2366", "value": True},
        {"id": "2367", "value": "AE"},
        {"id": "541", "value": "AE"},
    ]
    assert written_fields[3]["id"] == "2365"
    assert isinstance(written_fields[3]["value"], str)
    assert written_fields[3]["value"]
    # Result payload reports the written zone consistently.
    borrower = result["borrowers_processed"][0]
    assert borrower["in_flood_zone"] is True
    assert borrower["flood_zone"] == "AE"
    assert result["field_update"]["flood_zone"] == "AE"
    _assert_initial_processing_button_pressed(update_custom_fields, result)


@pytest.mark.asyncio
async def test_process_loan_writes_zone_x_when_not_in_flood_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unshaded property: Haiku returns flood_zone="" (no zone), sfha="no",
    # zone_found="yes". We write the 2366 boolean checkbox as False AND write
    # the 2367 and 541 dropdowns as "X" — FEMA Zone X is the minimal-hazard /
    # not-in-SFHA designation, so "not in a flood zone" is recorded as zone X
    # rather than left blank (and this overwrites any stale code from a prior run).
    _patch_common_process_loan_dependencies(monkeypatch)

    async def fake_capture(_address: dict[str, str]) -> tuple[bytes, int]:
        return b"%PDF-1.7\n", 9

    def fake_extract(_pdf_content: bytes) -> FloodZoneExtraction:
        return FloodZoneExtraction(success=True, flood_zone="", sfha="no", zone_found="yes")

    update_custom_fields = Mock()
    monkeypatch.setattr("ifd_agent.encompass_functions.capture_fema_pdf_async", fake_capture)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.upload_file_into_efolder",
        lambda *_a, **_k: "ok",
    )
    monkeypatch.setattr("ifd_agent.encompass_functions.extract_flood_zone", fake_extract)
    monkeypatch.setattr("ifd_agent.encompass_functions.update_custom_fields", update_custom_fields)

    result = await process_loan("12345", "IFD")

    # Not needs_review — empty zone with zone_found=yes is a confident
    # "not in any flood zone" outcome.
    assert result["status"] == "success"
    # Two field writes on the success path: the determination fields, then the
    # Initial Processing button press (the final Job Aid step).
    assert update_custom_fields.call_count == 2
    written_fields = update_custom_fields.call_args_list[0][0][4]
    assert written_fields[:3] == [
        {"id": "2366", "value": False},
        {"id": "2367", "value": "X"},
        {"id": "541", "value": "X"},
    ]
    assert written_fields[3]["id"] == "2365"
    assert isinstance(written_fields[3]["value"], str)
    assert written_fields[3]["value"]
    # Result payload reports the WRITTEN zone ("X"), not the raw extraction ("").
    borrower = result["borrowers_processed"][0]
    assert borrower["in_flood_zone"] is False
    assert borrower["flood_zone"] == "X"
    assert result["field_update"]["flood_zone"] == "X"
    # The raw extraction stays "" in the extraction payload (what Haiku saw).
    assert result["flood_zone_extraction"]["flood_zone"] == ""
    _assert_initial_processing_button_pressed(update_custom_fields, result)


@pytest.mark.asyncio
async def test_process_loan_clears_stale_ifd_comment_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A prior failed attempt left an IFD-authored "map unreadable" note in
    # Initial Processing Comments (CX.INITPROCNOTES). Once this attempt
    # completes successfully, the stale note must be cleared so the loan
    # doesn't look like it still has an unresolved flood-map problem.
    _patch_common_process_loan_dependencies(monkeypatch)

    async def fake_capture(_address: dict[str, str]) -> tuple[bytes, int]:
        return b"%PDF-1.7\n", 9

    def fake_extract(_pdf_content: bytes) -> FloodZoneExtraction:
        return FloodZoneExtraction(success=True, flood_zone="", sfha="no", zone_found="yes")

    update_custom_fields = Mock()
    monkeypatch.setattr("ifd_agent.encompass_functions.capture_fema_pdf_async", fake_capture)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.upload_file_into_efolder",
        lambda *_a, **_k: "ok",
    )
    monkeypatch.setattr("ifd_agent.encompass_functions.extract_flood_zone", fake_extract)
    monkeypatch.setattr("ifd_agent.encompass_functions.update_custom_fields", update_custom_fields)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.get_custom_field_value",
        lambda *_args, **_kwargs: "7/22 Image not printing for flood map on FEMA site",
    )

    result = await process_loan("12345", "IFD")

    assert result["status"] == "success"
    # Determination fields, button press, AND the stale-comment clear.
    assert update_custom_fields.call_count == 3
    clear_call = update_custom_fields.call_args_list[2][0][4]
    assert clear_call == [{"id": "CX.INITPROCNOTES", "value": ""}]


@pytest.mark.asyncio
async def test_process_loan_preserves_unrelated_ip_comment_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Initial Processing Comments is shared with the Prior Note/SECI/OFAC
    # agents (and human processors). A note IFD did not write itself — e.g.
    # an OFAC name-mismatch flag — must survive an IFD success untouched.
    _patch_common_process_loan_dependencies(monkeypatch)

    async def fake_capture(_address: dict[str, str]) -> tuple[bytes, int]:
        return b"%PDF-1.7\n", 9

    def fake_extract(_pdf_content: bytes) -> FloodZoneExtraction:
        return FloodZoneExtraction(success=True, flood_zone="", sfha="no", zone_found="yes")

    update_custom_fields = Mock()
    monkeypatch.setattr("ifd_agent.encompass_functions.capture_fema_pdf_async", fake_capture)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.upload_file_into_efolder",
        lambda *_a, **_k: "ok",
    )
    monkeypatch.setattr("ifd_agent.encompass_functions.extract_flood_zone", fake_extract)
    monkeypatch.setattr("ifd_agent.encompass_functions.update_custom_fields", update_custom_fields)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.get_custom_field_value",
        lambda *_args, **_kwargs: "Missing middle name - see OFAC review",
    )

    result = await process_loan("12345", "IFD")

    assert result["status"] == "success"
    # Only the determination fields and the button press — the unrelated
    # comment is never touched.
    assert update_custom_fields.call_count == 2


@pytest.mark.asyncio
async def test_process_loan_preserves_ip_comment_with_appended_text_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression test: a comment that combines IFD's own wording with text
    # appended by another agent/human must NOT be cleared. Only an EXACT
    # match to one of IFD's own templates is eligible — a substring check
    # would wrongly treat this mixed value as purely IFD-authored and erase
    # the appended (unrelated, still-relevant) note along with it.
    _patch_common_process_loan_dependencies(monkeypatch)

    async def fake_capture(_address: dict[str, str]) -> tuple[bytes, int]:
        return b"%PDF-1.7\n", 9

    def fake_extract(_pdf_content: bytes) -> FloodZoneExtraction:
        return FloodZoneExtraction(success=True, flood_zone="", sfha="no", zone_found="yes")

    update_custom_fields = Mock()
    monkeypatch.setattr("ifd_agent.encompass_functions.capture_fema_pdf_async", fake_capture)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.upload_file_into_efolder",
        lambda *_a, **_k: "ok",
    )
    monkeypatch.setattr("ifd_agent.encompass_functions.extract_flood_zone", fake_extract)
    monkeypatch.setattr("ifd_agent.encompass_functions.update_custom_fields", update_custom_fields)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.get_custom_field_value",
        lambda *_args, **_kwargs: (
            "7/22 Image not printing for flood map on FEMA site; also missing MI - OFAC"
        ),
    )

    result = await process_loan("12345", "IFD")

    assert result["status"] == "success"
    assert update_custom_fields.call_count == 2


def test_is_ifd_authored_ip_comment_requires_exact_match() -> None:
    # Exact IFD templates are recognized...
    assert _is_ifd_authored_ip_comment("No digital data available") is True
    assert _is_ifd_authored_ip_comment("6/18 Image not printing for flood map on FEMA site") is True
    assert _is_ifd_authored_ip_comment("12/3 Image not printing for flood map on FEMA site") is True
    # ...but any deviation — appended text, different wording, empty, or a
    # wholly unrelated note — is not, so it's never cleared.
    assert _is_ifd_authored_ip_comment("") is False
    assert _is_ifd_authored_ip_comment("Missing middle name - see OFAC review") is False
    assert (
        _is_ifd_authored_ip_comment(
            "6/18 Image not printing for flood map on FEMA site; also missing MI - OFAC"
        )
        is False
    )
    assert _is_ifd_authored_ip_comment("No digital data available for this property") is False


@pytest.mark.asyncio
async def test_process_loan_zone_x_is_not_in_flood_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Orange / 0.2%-annual-chance area: Haiku returns flood_zone="X". Zone X is
    # NOT an SFHA, so 2366 must be False even though a zone code is present —
    # and even if the vision model's sfha guess wrongly says "yes" (the
    # deterministic zone-code classification is authoritative). 2367/541 = "X".
    _patch_common_process_loan_dependencies(monkeypatch)

    async def fake_capture(_address: dict[str, str]) -> tuple[bytes, int]:
        return b"%PDF-1.7\n", 9

    def fake_extract(_pdf_content: bytes) -> FloodZoneExtraction:
        # sfha="yes" is deliberately wrong to prove the zone code wins.
        return FloodZoneExtraction(success=True, flood_zone="X", sfha="yes", zone_found="yes")

    update_custom_fields = Mock()
    monkeypatch.setattr("ifd_agent.encompass_functions.capture_fema_pdf_async", fake_capture)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.upload_file_into_efolder",
        lambda *_a, **_k: "ok",
    )
    monkeypatch.setattr("ifd_agent.encompass_functions.extract_flood_zone", fake_extract)
    monkeypatch.setattr("ifd_agent.encompass_functions.update_custom_fields", update_custom_fields)

    result = await process_loan("12345", "IFD")

    assert result["status"] == "success"
    written_fields = update_custom_fields.call_args_list[0][0][4]
    assert written_fields[:3] == [
        {"id": "2366", "value": False},
        {"id": "2367", "value": "X"},
        {"id": "541", "value": "X"},
    ]
    borrower = result["borrowers_processed"][0]
    assert borrower["in_flood_zone"] is False
    assert borrower["flood_zone"] == "X"


@pytest.mark.asyncio
async def test_process_loan_named_non_sfha_zone_d_preserves_code_and_is_not_in_flood_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Zone D (undetermined risk) is a named NON-SFHA zone. 2366 must be False,
    # and the true code "D" must be recorded in 2367/541 — NOT flattened to "X".
    _patch_common_process_loan_dependencies(monkeypatch)

    async def fake_capture(_address: dict[str, str]) -> tuple[bytes, int]:
        return b"%PDF-1.7\n", 9

    def fake_extract(_pdf_content: bytes) -> FloodZoneExtraction:
        return FloodZoneExtraction(success=True, flood_zone="D", sfha="no", zone_found="yes")

    update_custom_fields = Mock()
    monkeypatch.setattr("ifd_agent.encompass_functions.capture_fema_pdf_async", fake_capture)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.upload_file_into_efolder",
        lambda *_a, **_k: "ok",
    )
    monkeypatch.setattr("ifd_agent.encompass_functions.extract_flood_zone", fake_extract)
    monkeypatch.setattr("ifd_agent.encompass_functions.update_custom_fields", update_custom_fields)

    result = await process_loan("12345", "IFD")

    assert result["status"] == "success"
    written_fields = update_custom_fields.call_args_list[0][0][4]
    assert written_fields[:3] == [
        {"id": "2366", "value": False},
        {"id": "2367", "value": "D"},
        {"id": "541", "value": "D"},
    ]
    borrower = result["borrowers_processed"][0]
    assert borrower["in_flood_zone"] is False
    assert borrower["flood_zone"] == "D"


@pytest.mark.asyncio
async def test_process_loan_partial_when_button_press_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Determination fields and upload succeed, but the final button press
    # fails. The run must NOT report "success" — it drops to "partial" so the
    # loan is re-picked rather than silently left without its completion stamp.
    _patch_common_process_loan_dependencies(monkeypatch)

    async def fake_capture(_address: dict[str, str]) -> tuple[bytes, int]:
        return b"%PDF-1.7\n", 9

    def fake_extract(_pdf_content: bytes) -> FloodZoneExtraction:
        return FloodZoneExtraction(success=True, flood_zone="AE", sfha="yes", zone_found="yes")

    # First call (determination fields) succeeds; second call (button) raises.
    update_custom_fields = Mock(side_effect=[None, RuntimeError("fieldWriter 500")])
    monkeypatch.setattr("ifd_agent.encompass_functions.capture_fema_pdf_async", fake_capture)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.upload_file_into_efolder",
        lambda *_a, **_k: "ok",
    )
    monkeypatch.setattr("ifd_agent.encompass_functions.extract_flood_zone", fake_extract)
    monkeypatch.setattr("ifd_agent.encompass_functions.update_custom_fields", update_custom_fields)

    result = await process_loan("12345", "IFD")

    assert update_custom_fields.call_count == 2
    assert result["status"] == "partial"
    assert result["initial_processing_button"]["pressed"] is False
    assert "fieldWriter 500" in result["initial_processing_button"]["error"]


@pytest.mark.asyncio
async def test_process_loan_no_digital_data_writes_ip_comment_and_skips_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No digital data available: Haiku sets no_digital_data="yes". We must
    # write ONLY the Initial Processing Comments field ("No digital data
    # available"), NOT upload the PDF, NOT write any determination field, and
    # return error so the run surfaces as Errored (needs-review deferred,
    # BRAVO-135).
    _patch_common_process_loan_dependencies(monkeypatch)

    async def fake_capture(_address: dict[str, str]) -> tuple[bytes, int]:
        return b"%PDF-1.7\n", 9

    def fake_extract(_pdf_content: bytes) -> FloodZoneExtraction:
        return FloodZoneExtraction(
            success=True,
            flood_zone="",
            sfha="no",
            zone_found="no",
            no_digital_data="yes",
        )

    upload_file_into_efolder = Mock(return_value="should-not-be-called")
    update_custom_fields = Mock()
    monkeypatch.setattr("ifd_agent.encompass_functions.capture_fema_pdf_async", fake_capture)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.upload_file_into_efolder", upload_file_into_efolder
    )
    monkeypatch.setattr("ifd_agent.encompass_functions.extract_flood_zone", fake_extract)
    monkeypatch.setattr("ifd_agent.encompass_functions.update_custom_fields", update_custom_fields)

    result = await process_loan("12345", "IFD")

    assert result["status"] == "error"
    # PDF upload must be skipped entirely.
    upload_file_into_efolder.assert_not_called()
    borrower = result["borrowers_processed"][0]
    assert borrower["efolder_uploaded"] is False
    assert borrower["no_usable_map"] is True
    # A PDF was captured but deliberately NOT filed — so map_captured is True
    # yet pdf_filename (which signals "filed to eFolder") must be absent.
    assert borrower["map_captured"] is True
    assert "pdf_filename" not in borrower
    # The ONLY field write is the IP comment — no 2366/2367/541/2365.
    update_custom_fields.assert_called_once()
    written_fields = update_custom_fields.call_args[0][4]
    assert written_fields == [{"id": "CX.INITPROCNOTES", "value": "No digital data available"}]


@pytest.mark.asyncio
async def test_process_loan_map_unreadable_writes_ip_comment_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Corrupted/garbled FEMA map render: Haiku sets map_unreadable="yes". We
    # must write ONLY the dated Initial Processing Comments note, NOT upload the
    # PDF, NOT write any determination field, NOT press the Initial Processing
    # button, and return status="error" so the run surfaces as Errored (and is
    # retryable for a fresh capture next pass).
    _patch_common_process_loan_dependencies(monkeypatch)

    async def fake_capture(_address: dict[str, str]) -> tuple[bytes, int]:
        return b"%PDF-1.7\n", 9

    def fake_extract(_pdf_content: bytes) -> FloodZoneExtraction:
        return FloodZoneExtraction(
            success=True,
            flood_zone="",
            sfha="unclear",
            zone_found="no",
            no_digital_data="no",
            map_unreadable="yes",
        )

    upload_file_into_efolder = Mock(return_value="should-not-be-called")
    update_custom_fields = Mock()
    monkeypatch.setattr("ifd_agent.encompass_functions.capture_fema_pdf_async", fake_capture)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.upload_file_into_efolder", upload_file_into_efolder
    )
    monkeypatch.setattr("ifd_agent.encompass_functions.extract_flood_zone", fake_extract)
    monkeypatch.setattr("ifd_agent.encompass_functions.update_custom_fields", update_custom_fields)

    result = await process_loan("12345", "IFD")

    assert result["status"] == "error"
    # PDF upload must be skipped entirely.
    upload_file_into_efolder.assert_not_called()
    borrower = result["borrowers_processed"][0]
    assert borrower["map_unreadable"] is True
    assert borrower["map_captured"] is True
    assert borrower["efolder_uploaded"] is False
    assert "pdf_filename" not in borrower
    # The ONLY field write is the dated IP comment — no 2366/2367/541/2365 and no
    # Initial Processing button (CX.INITIAL.FLOOD.DETER.*).
    update_custom_fields.assert_called_once()
    written_fields = update_custom_fields.call_args[0][4]
    assert len(written_fields) == 1
    assert written_fields[0]["id"] == "CX.INITPROCNOTES"
    note = written_fields[0]["value"]
    assert note.endswith("Image not printing for flood map on FEMA site")
    assert "/" in note.split(" ", 1)[0]  # dated prefix like "6/18"


@pytest.mark.asyncio
async def test_map_unreadable_comment_date_prefix_is_client_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The M/D prefix on the IP comment is read by staff, so it must come from
    # the client's clock. Pinned to 9:30 PM Mountain, which is already 8/1 in
    # UTC — a UTC clock would write "8/1" while staff are still on 7/31.
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from ifd_agent.encompass_functions import MAP_UNREADABLE_COMMENT_BASE

    _patch_common_process_loan_dependencies(monkeypatch)

    async def fake_capture(_address: dict[str, str]) -> tuple[bytes, int]:
        return b"%PDF-1.7\n", 9

    def fake_extract(_pdf_content: bytes) -> FloodZoneExtraction:
        return FloodZoneExtraction(
            success=True,
            flood_zone="",
            sfha="unclear",
            zone_found="no",
            no_digital_data="no",
            map_unreadable="yes",
        )

    update_custom_fields = Mock()
    monkeypatch.setattr("ifd_agent.encompass_functions.capture_fema_pdf_async", fake_capture)
    monkeypatch.setattr("ifd_agent.encompass_functions.extract_flood_zone", fake_extract)
    monkeypatch.setattr("ifd_agent.encompass_functions.update_custom_fields", update_custom_fields)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.client_now",
        lambda: datetime(2026, 7, 31, 21, 30, tzinfo=ZoneInfo("America/Denver")),
    )

    await process_loan("12345", "IFD")

    written_fields = update_custom_fields.call_args[0][4]
    assert written_fields[0]["value"] == f"7/31 {MAP_UNREADABLE_COMMENT_BASE}"


@pytest.mark.asyncio
async def test_process_loan_deterministic_capture_failure_routes_to_manual_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A deterministic capture failure (e.g. the NFHL viewer never produced a
    # search box) is a "no usable map" outcome. We write the IP comment, skip
    # the upload entirely, never run extraction, and surface the run as Errored
    # (needs-review deferred, BRAVO-135).
    _patch_common_process_loan_dependencies(monkeypatch)

    async def fake_capture(_address: dict[str, str]) -> tuple[bytes, int]:
        raise RuntimeError("Could not locate NFHL search input on viewer page")

    upload_file_into_efolder = Mock(return_value="should-not-be-called")
    extract_flood_zone = Mock()
    update_custom_fields = Mock()
    monkeypatch.setattr("ifd_agent.encompass_functions.capture_fema_pdf_async", fake_capture)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.upload_file_into_efolder", upload_file_into_efolder
    )
    monkeypatch.setattr("ifd_agent.encompass_functions.extract_flood_zone", extract_flood_zone)
    monkeypatch.setattr("ifd_agent.encompass_functions.update_custom_fields", update_custom_fields)

    result = await process_loan("12345", "IFD")

    assert result["status"] == "error"
    upload_file_into_efolder.assert_not_called()
    extract_flood_zone.assert_not_called()
    borrower = result["borrowers_processed"][0]
    assert borrower["no_usable_map"] is True
    assert borrower["efolder_uploaded"] is False
    assert "pdf_filename" not in borrower  # no PDF was captured
    update_custom_fields.assert_called_once()
    written_fields = update_custom_fields.call_args[0][4]
    assert written_fields == [{"id": "CX.INITPROCNOTES", "value": "No digital data available"}]


@pytest.mark.asyncio
async def test_process_loan_transient_capture_failure_stays_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A transient capture failure (navigation timeout) must NOT be swallowed
    # into manual review — it stays a retryable `error` so the orchestrator can
    # try again. No IP comment is written for the transient case.
    _patch_common_process_loan_dependencies(monkeypatch)

    async def fake_capture(_address: dict[str, str]) -> tuple[bytes, int]:
        raise RuntimeError("Navigation timeout of 30000 ms exceeded")

    update_custom_fields = Mock()
    monkeypatch.setattr("ifd_agent.encompass_functions.capture_fema_pdf_async", fake_capture)
    monkeypatch.setattr("ifd_agent.encompass_functions.update_custom_fields", update_custom_fields)

    result = await process_loan("12345", "IFD")

    assert result["status"] == "error"
    update_custom_fields.assert_not_called()


@pytest.mark.asyncio
async def test_process_loan_skips_pdf_upload_on_extraction_failure_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Extraction failure (e.g. Bedrock ValidationException) surfaces as errored
    # (needs-review deferred, BRAVO-135) and must not file the captured PDF.
    _patch_common_process_loan_dependencies(monkeypatch)

    async def fake_capture(_address: dict[str, str]) -> tuple[bytes, int]:
        return b"%PDF-1.7\n", 9

    def fake_extract(_pdf_content: bytes) -> FloodZoneExtraction:
        return FloodZoneExtraction(
            success=False,
            error="bedrock_client_error:validation_error",
            zone_found="no",
        )

    upload_file_into_efolder = Mock(side_effect=RuntimeError("should-not-be-called"))
    update_custom_fields = Mock()
    monkeypatch.setattr("ifd_agent.encompass_functions.capture_fema_pdf_async", fake_capture)
    monkeypatch.setattr(
        "ifd_agent.encompass_functions.upload_file_into_efolder",
        upload_file_into_efolder,
    )
    monkeypatch.setattr("ifd_agent.encompass_functions.extract_flood_zone", fake_extract)
    monkeypatch.setattr("ifd_agent.encompass_functions.update_custom_fields", update_custom_fields)

    result = await process_loan("12345", "IFD")

    assert result["status"] == "error"
    upload_file_into_efolder.assert_not_called()
    borrower = result["borrowers_processed"][0]
    assert borrower["efolder_uploaded"] is False
    assert borrower["map_captured"] is True
    assert "pdf_filename" not in borrower
    assert result["flood_zone_extraction"]["error"] == "bedrock_client_error:validation_error"
    update_custom_fields.assert_not_called()


def test_emit_step_events_uses_upload_status_without_double_prefix() -> None:
    tool = EncompassMCPTool()
    tracker = _FakeTracker()
    result = {
        "status": "partial",
        "address": {"street": "6123 Raleigh St", "city": "Spring Hill"},
        "borrowers_processed": [
            {
                "search_url": "https://msc.fema.gov/portal/home",
                "pdf_filename": "FLOODSEARCH.pdf",
                "efolder_uploaded": False,
                "status": "Upload failed: boom",
            }
        ],
        "flood_zone_extraction": {"zone_found": "yes", "flood_zone": "AE", "sfha": "yes"},
    }

    tool._emit_step_events(tracker, result, status_lower="partial", is_success=True)

    assert ("error", STEP_UPLOAD_EFOLDER_IFD, "Upload failed: boom") in tracker.events
    assert (
        "error",
        STEP_UPLOAD_EFOLDER_IFD,
        "Upload failed: Upload failed: boom",
    ) not in tracker.events
    upload_idx = tracker.events.index(("error", STEP_UPLOAD_EFOLDER_IFD, "Upload failed: boom"))
    extraction_idx = next(
        index for index, event in enumerate(tracker.events) if event[1] == STEP_EXTRACT_FLOOD_ZONE
    )
    assert upload_idx < extraction_idx


def test_emit_step_events_records_vision_low_confidence_as_errored() -> None:
    # A low-confidence extraction is surfaced as errored (needs-review deferred,
    # BRAVO-135): the vision step is errored, and the upload step is NOT emitted
    # (we intentionally didn't file the PDF).
    tool = EncompassMCPTool()
    tracker = _FakeTracker()
    result = {
        "status": "error",
        "address": {"street": "10905 Claywood Dr", "city": "Austin"},
        "borrowers_processed": [
            {
                "search_url": "https://hazards-fema.maps.arcgis.com/...",
                "map_captured": True,
                "efolder_uploaded": False,
                "status": "Flood zone could not be classified",
            }
        ],
        "flood_zone_extraction": {"zone_found": "no", "error": "low_confidence"},
    }

    tool._emit_step_events(tracker, result, status_lower="error", is_success=False)

    # The vision step is errored (the stopping point).
    errored_vision = [
        event
        for event in tracker.events
        if event[1] == STEP_EXTRACT_FLOOD_ZONE and event[0] == "error"
    ]
    assert len(errored_vision) == 1
    # The upload step is never emitted (we intentionally didn't upload).
    upload_events = [event for event in tracker.events if event[1] == STEP_UPLOAD_EFOLDER_IFD]
    assert upload_events == []


def test_emit_step_events_no_usable_map_with_pdf_errors_vision_not_upload() -> None:
    # No-usable-map where a PDF WAS captured (FEMA no-data / blank render).
    # The vision step is errored (stopping point); the upload step must NOT be
    # emitted (we intentionally didn't file the PDF).
    tool = EncompassMCPTool()
    tracker = _FakeTracker()
    result = {
        "status": "error",
        "address": {"street": "1478 Riviera Ave", "city": "New Orleans"},
        "borrowers_processed": [
            {
                "search_url": "https://hazards-fema.maps.arcgis.com/...",
                "map_captured": True,
                "efolder_uploaded": False,
                "status": "No digital data available",
                "no_usable_map": True,
            }
        ],
        "flood_zone_extraction": {"zone_found": "no", "no_digital_data": "yes"},
    }

    tool._emit_step_events(tracker, result, status_lower="error", is_success=False)

    # Vision step is errored; the upload step is the one that must NOT appear.
    errored_events = [event for event in tracker.events if event[0] == "error"]
    assert any(event[1] == STEP_EXTRACT_FLOOD_ZONE for event in errored_events)
    upload_events = [event for event in tracker.events if event[1] == STEP_UPLOAD_EFOLDER_IFD]
    assert upload_events == []


def test_emit_step_events_no_usable_map_capture_failure_errors_capture() -> None:
    # No-usable-map from a deterministic CAPTURE failure — no PDF was produced,
    # so there's no extraction. The stopping point is the capture step, which
    # is errored, and no upload step is emitted.
    tool = EncompassMCPTool()
    tracker = _FakeTracker()
    result = {
        "status": "error",
        "address": {"street": "1478 Riviera Ave", "city": "New Orleans"},
        "borrowers_processed": [
            {
                "search_url": "https://hazards-fema.maps.arcgis.com/...",
                "efolder_uploaded": False,
                "status": "No digital data available",
                "no_usable_map": True,
            }
        ],
    }

    tool._emit_step_events(tracker, result, status_lower="error", is_success=False)

    # Capture step is errored (the stopping point); no upload step emitted.
    errored_events = [event for event in tracker.events if event[0] == "error"]
    assert any(event[1] == STEP_CAPTURE_PDF_FEMA for event in errored_events)
    assert [event for event in tracker.events if event[1] == STEP_UPLOAD_EFOLDER_IFD] == []


def test_initial_processing_fields_use_iso_date() -> None:
    # CX.INITIAL.FLOOD.DETER.DATE is a date-typed Encompass field that requires
    # ISO yyyy-MM-dd; sending US MM/dd/yyyy fails with a Serialization 400 and
    # leaves the Initial Processing button unpressed. Lock the format in.
    import re
    from datetime import datetime

    from ifd_agent.encompass_functions import (
        INITIAL_PROC_FLOOD_DATE_FIELD_ID,
        INITIAL_PROC_FLOOD_USER_FIELD_ID,
        _build_initial_processing_fields,
    )

    by_id = {f["id"]: f["value"] for f in _build_initial_processing_fields("dashapi")}

    assert by_id[INITIAL_PROC_FLOOD_USER_FIELD_ID] == "dashapi"
    date_value = by_id[INITIAL_PROC_FLOOD_DATE_FIELD_ID]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_value), date_value
    # Parseable as a real ISO date (no timezone offset).
    datetime.strptime(date_value, "%Y-%m-%d")


def test_initial_processing_fields_are_client_local_not_utc() -> None:
    # The flood button's date/time halves are read on the Initial Processing
    # screen, so they must carry the client's wall clock rather than the
    # container's UTC clock (a 6-7 hour error, and a date rollover at night).
    # Pinned to 9:30 PM Mountain, which is already the *next* day in UTC
    # (2026-08-01T03:30:15Z) — so a UTC stamp would fail on the date alone.
    from datetime import datetime
    from unittest.mock import patch
    from zoneinfo import ZoneInfo

    from ifd_agent.encompass_functions import (
        INITIAL_PROC_FLOOD_DATE_FIELD_ID,
        INITIAL_PROC_FLOOD_TIME_FIELD_ID,
        _build_initial_processing_fields,
    )

    pinned = datetime(2026, 7, 31, 21, 30, 15, tzinfo=ZoneInfo("America/Denver"))
    with patch("ifd_agent.encompass_functions.client_now", return_value=pinned):
        by_id = {f["id"]: f["value"] for f in _build_initial_processing_fields("dashapi")}

    assert by_id[INITIAL_PROC_FLOOD_DATE_FIELD_ID] == "2026-07-31"
    assert by_id[INITIAL_PROC_FLOOD_TIME_FIELD_ID] == "09:30:15 PM"


@pytest.mark.parametrize(
    ("zone", "expected_sfha"),
    [
        # SFHA (in a flood zone) — the A*/V* family.
        ("A", True),
        ("AE", True),
        ("AH", True),
        ("AO", True),
        ("A99", True),
        ("AR", True),
        ("A12", True),
        ("V", True),
        ("VE", True),
        ("V30", True),
        ("ae", True),  # case-insensitive
        # NOT SFHA — X (incl. orange/0.2% shaded and X500), B, C, D, empty.
        ("X", False),
        ("X500", False),
        ("B", False),
        ("C", False),
        ("D", False),
        ("", False),
        ("   ", False),
    ],
)
def test_is_sfha_zone(zone: str, expected_sfha: bool) -> None:
    assert _is_sfha_zone(zone) is expected_sfha
