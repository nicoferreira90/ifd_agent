"""Smoke tests for the Bedrock-based flood zone extractor.

These tests do not call Bedrock at all. They install a fake `bedrock-runtime`
client that returns a canned Converse response and patch out PyMuPDF so we
exercise the response-parsing code paths only.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from ifd_agent.vision_assistant import flood_zone_extractor


class _FakeBedrockClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.response


def _stub_render(monkeypatch: pytest.MonkeyPatch, png_count: int = 1) -> None:
    """Replace the PyMuPDF renderer so tests don't need real PDF input."""

    def _fake_render(_pdf_bytes: bytes, *, max_pages: int = 2, dpi: int = 150) -> list[bytes]:
        return [b"\x89PNG\r\n\x1a\n"] * png_count

    monkeypatch.setattr(flood_zone_extractor, "_render_pdf_pages_to_png", _fake_render)


def test_extract_flood_zone_success(
    monkeypatch: pytest.MonkeyPatch,
    fake_bedrock_extraction_response: dict[str, Any],
) -> None:
    _stub_render(monkeypatch)
    client = _FakeBedrockClient(fake_bedrock_extraction_response)

    extraction = flood_zone_extractor.extract_flood_zone(b"%PDF-fake-bytes", bedrock_client=client)

    assert extraction.success
    assert extraction.flood_zone == "X"
    assert extraction.sfha == "no"
    assert extraction.zone_found == "yes"
    assert extraction.panel_number == "12345C0123F"
    assert extraction.community_id == "120000"
    assert len(client.calls) == 1
    assert client.calls[0]["modelId"]
    assert client.calls[0]["toolConfig"]["toolChoice"]["tool"]["name"] == "extract_flood_zone"


def test_extract_flood_zone_handles_missing_tool_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_render(monkeypatch)
    client = _FakeBedrockClient({"output": {"message": {"content": [{"text": "no tool here"}]}}})

    extraction = flood_zone_extractor.extract_flood_zone(b"%PDF-fake-bytes", bedrock_client=client)

    assert not extraction.success
    assert extraction.error is not None
    assert extraction.error.startswith("malformed_tool_use_response")


def test_extract_flood_zone_short_circuits_on_render_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_pdf_bytes: bytes, *, max_pages: int = 2, dpi: int = 150) -> list[bytes]:
        raise ValueError("empty_pdf_bytes")

    monkeypatch.setattr(flood_zone_extractor, "_render_pdf_pages_to_png", _boom)

    extraction = flood_zone_extractor.extract_flood_zone(b"", bedrock_client=MagicMock())

    assert not extraction.success
    assert extraction.error == "render_failed:empty_pdf_bytes"


def test_extract_flood_zone_flags_low_confidence_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_render(monkeypatch)
    client = _FakeBedrockClient(
        {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "extract_flood_zone",
                                "input": {
                                    "flood_zone": "",
                                    "sfha": "unclear",
                                    "panel_number": "",
                                    "panel_effective_date": "",
                                    "community_name": "",
                                    "community_id": "",
                                    "zone_found": "no",
                                    "no_digital_data": "no",
                                    "map_unreadable": "no",
                                },
                            }
                        }
                    ]
                }
            }
        }
    )

    extraction = flood_zone_extractor.extract_flood_zone(b"%PDF-fake-bytes", bedrock_client=client)

    assert extraction.success
    assert extraction.zone_found == "no"
    assert extraction.flood_zone == ""
    assert not extraction.has_primary_fields()


def test_extract_flood_zone_accepts_unshaded_no_zone_as_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An unshaded property is a *confident* "not in any flood zone" outcome —
    # Haiku returns flood_zone="", sfha="no", zone_found="yes" per the prompt.
    # The extractor must accept this as valid (success=True, primary fields
    # OK) so the downstream branch can write the 2366 checkbox as "N" rather
    # than routing to needs_review.
    _stub_render(monkeypatch)
    client = _FakeBedrockClient(
        {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "extract_flood_zone",
                                "input": {
                                    "flood_zone": "",
                                    "sfha": "no",
                                    "panel_number": "12053C0303E",
                                    "panel_effective_date": "1/15/2021",
                                    "community_name": "HERNANDO COUNTY",
                                    "community_id": "",
                                    "zone_found": "yes",
                                    "no_digital_data": "no",
                                    "map_unreadable": "no",
                                },
                            }
                        }
                    ]
                }
            }
        }
    )

    extraction = flood_zone_extractor.extract_flood_zone(b"%PDF-fake-bytes", bedrock_client=client)

    assert extraction.success
    assert extraction.zone_found == "yes"
    assert extraction.flood_zone == ""
    assert extraction.sfha == "no"
    assert extraction.has_primary_fields()  # ← the new behavior


def test_extract_flood_zone_flags_no_digital_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # When the FEMA map has no digital flood data for the property, Haiku
    # sets no_digital_data="yes". The extractor must surface that via the
    # `is_no_digital_data` property so the pipeline can route to manual
    # review without uploading a PDF or writing determination fields.
    _stub_render(monkeypatch)
    client = _FakeBedrockClient(
        {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "extract_flood_zone",
                                "input": {
                                    "flood_zone": "",
                                    "sfha": "no",
                                    "panel_number": "",
                                    "panel_effective_date": "",
                                    "community_name": "",
                                    "community_id": "",
                                    "zone_found": "no",
                                    "no_digital_data": "yes",
                                    "map_unreadable": "no",
                                },
                            }
                        }
                    ]
                }
            }
        }
    )

    extraction = flood_zone_extractor.extract_flood_zone(b"%PDF-fake-bytes", bedrock_client=client)

    assert extraction.success
    assert extraction.is_no_digital_data is True


def test_extract_flood_zone_no_digital_data_defaults_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A normal classification (unshaded property with full map data) must
    # NOT be treated as no-digital-data — that's the most common case and
    # the two must never be conflated.
    _stub_render(monkeypatch)
    client = _FakeBedrockClient(
        {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "extract_flood_zone",
                                "input": {
                                    "flood_zone": "",
                                    "sfha": "no",
                                    "panel_number": "48453C0460K",
                                    "panel_effective_date": "1/6/2016",
                                    "community_name": "Austin",
                                    "community_id": "",
                                    "zone_found": "yes",
                                    "no_digital_data": "no",
                                    "map_unreadable": "no",
                                },
                            }
                        }
                    ]
                }
            }
        }
    )

    extraction = flood_zone_extractor.extract_flood_zone(b"%PDF-fake-bytes", bedrock_client=client)

    assert extraction.success
    assert extraction.is_no_digital_data is False
    assert extraction.is_map_unreadable is False
    assert extraction.has_primary_fields()


def test_extract_flood_zone_flags_map_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # When the FEMA map rendered but is visually corrupted (garbled crosshatch
    # mesh), Haiku sets map_unreadable="yes". The extractor must surface that
    # via `is_map_unreadable` so the pipeline writes the IP comment and
    # surfaces the run as an error — distinct from the no-digital-data case.
    _stub_render(monkeypatch)
    client = _FakeBedrockClient(
        {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "extract_flood_zone",
                                "input": {
                                    "flood_zone": "",
                                    "sfha": "unclear",
                                    "panel_number": "",
                                    "panel_effective_date": "",
                                    "community_name": "",
                                    "community_id": "",
                                    "zone_found": "no",
                                    "no_digital_data": "no",
                                    "map_unreadable": "yes",
                                },
                            }
                        }
                    ]
                }
            }
        }
    )

    extraction = flood_zone_extractor.extract_flood_zone(b"%PDF-fake-bytes", bedrock_client=client)

    assert extraction.success
    assert extraction.is_map_unreadable is True
    assert extraction.is_no_digital_data is False


def test_extract_flood_zone_rejects_inconsistent_zone_yes_with_empty_sfha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # zone_found=yes with empty flood_zone AND empty sfha is incoherent —
    # the model claimed confidence but didn't classify either field. This
    # must still fail validation so the loan goes to needs_review.
    _stub_render(monkeypatch)
    client = _FakeBedrockClient(
        {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "extract_flood_zone",
                                "input": {
                                    "flood_zone": "",
                                    "sfha": "",
                                    "panel_number": "",
                                    "panel_effective_date": "",
                                    "community_name": "",
                                    "community_id": "",
                                    "zone_found": "yes",
                                    "no_digital_data": "no",
                                    "map_unreadable": "no",
                                },
                            }
                        }
                    ]
                }
            }
        }
    )

    extraction = flood_zone_extractor.extract_flood_zone(b"%PDF-fake-bytes", bedrock_client=client)

    assert extraction.success is False
    assert extraction.error == "invalid_primary_fields"


def test_extract_flood_zone_rejects_zone_ae_with_sfha_no(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Zone AE is unambiguously SFHA per FEMA — if the model claims AE but
    # reports sfha="no", the two contradict each other. The model is
    # confused about one of the fields; we don't know which, so route to
    # needs_review rather than auto-writing.
    _stub_render(monkeypatch)
    client = _FakeBedrockClient(
        {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "extract_flood_zone",
                                "input": {
                                    "flood_zone": "AE",
                                    "sfha": "no",
                                    "panel_number": "12053C0303E",
                                    "panel_effective_date": "1/15/2021",
                                    "community_name": "HERNANDO COUNTY",
                                    "community_id": "",
                                    "zone_found": "yes",
                                    "no_digital_data": "no",
                                    "map_unreadable": "no",
                                },
                            }
                        }
                    ]
                }
            }
        }
    )

    extraction = flood_zone_extractor.extract_flood_zone(b"%PDF-fake-bytes", bedrock_client=client)

    assert extraction.success is False
    assert extraction.error == "invalid_primary_fields"


def test_extract_flood_zone_rejects_zone_x_with_sfha_yes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Zone X is unambiguously NOT SFHA — sfha="yes" contradicts the zone.
    # Same logic as AE+no: the model contradicted itself, route to review.
    _stub_render(monkeypatch)
    client = _FakeBedrockClient(
        {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "extract_flood_zone",
                                "input": {
                                    "flood_zone": "X",
                                    "sfha": "yes",
                                    "panel_number": "",
                                    "panel_effective_date": "",
                                    "community_name": "",
                                    "community_id": "",
                                    "zone_found": "yes",
                                    "no_digital_data": "no",
                                    "map_unreadable": "no",
                                },
                            }
                        }
                    ]
                }
            }
        }
    )

    extraction = flood_zone_extractor.extract_flood_zone(b"%PDF-fake-bytes", bedrock_client=client)

    assert extraction.success is False
    assert extraction.error == "invalid_primary_fields"


def test_extract_flood_zone_normalizes_lowercase_zone_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Model occasionally emits "ae" or "Ae" instead of canonical "AE".
    # The extractor must canonicalize at parse time so the value that
    # ultimately lands in Encompass field 2367 is the strict code the
    # dropdown expects.
    _stub_render(monkeypatch)
    client = _FakeBedrockClient(
        {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "extract_flood_zone",
                                "input": {
                                    "flood_zone": "ae",
                                    "sfha": "yes",
                                    "panel_number": "12053C0303E",
                                    "panel_effective_date": "1/15/2021",
                                    "community_name": "HERNANDO COUNTY",
                                    "community_id": "",
                                    "zone_found": "yes",
                                    "no_digital_data": "no",
                                    "map_unreadable": "no",
                                },
                            }
                        }
                    ]
                }
            }
        }
    )

    extraction = flood_zone_extractor.extract_flood_zone(b"%PDF-fake-bytes", bedrock_client=client)

    assert extraction.success
    assert extraction.flood_zone == "AE"  # canonicalized
    assert extraction.has_primary_fields()


def test_extract_flood_zone_normalizes_zone_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Model sometimes emits "Zone AE" (mirroring how the value is labeled
    # on the FEMA map) instead of just "AE". The extractor must strip the
    # prefix so Encompass receives the bare code.
    _stub_render(monkeypatch)
    client = _FakeBedrockClient(
        {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "extract_flood_zone",
                                "input": {
                                    "flood_zone": "Zone AE",
                                    "sfha": "yes",
                                    "panel_number": "",
                                    "panel_effective_date": "",
                                    "community_name": "",
                                    "community_id": "",
                                    "zone_found": "yes",
                                    "no_digital_data": "no",
                                    "map_unreadable": "no",
                                },
                            }
                        }
                    ]
                }
            }
        }
    )

    extraction = flood_zone_extractor.extract_flood_zone(b"%PDF-fake-bytes", bedrock_client=client)

    assert extraction.success
    assert extraction.flood_zone == "AE"  # "Zone " prefix stripped
    assert extraction.has_primary_fields()


def test_extract_flood_zone_accepts_unrecognized_zone_with_definite_sfha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unrecognized zone codes (outside FEMA's known A/V/B/C/D/X set) must
    # NOT be over-rejected — we accept them as long as sfha is a definite
    # yes/no. This is a safety net for rare or future zone variants.
    _stub_render(monkeypatch)
    client = _FakeBedrockClient(
        {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "extract_flood_zone",
                                "input": {
                                    "flood_zone": "ZZ99",  # made-up but treated as opaque
                                    "sfha": "yes",
                                    "panel_number": "",
                                    "panel_effective_date": "",
                                    "community_name": "",
                                    "community_id": "",
                                    "zone_found": "yes",
                                    "no_digital_data": "no",
                                    "map_unreadable": "no",
                                },
                            }
                        }
                    ]
                }
            }
        }
    )

    extraction = flood_zone_extractor.extract_flood_zone(b"%PDF-fake-bytes", bedrock_client=client)

    assert extraction.success
    assert extraction.has_primary_fields()


def test_extract_flood_zone_rejects_zone_code_with_unclear_sfha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Per the vision prompt, sfha must be a definite "yes" or "no" — never
    # "unclear" — when a zone code is picked under zone_found="yes". If the
    # model violates that (picks a zone but reports unclear SFHA), the
    # extraction is insufficiently confident and must route to needs_review
    # rather than auto-writing the dropdown.
    _stub_render(monkeypatch)
    client = _FakeBedrockClient(
        {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "extract_flood_zone",
                                "input": {
                                    "flood_zone": "AE",
                                    "sfha": "unclear",
                                    "panel_number": "12053C0303E",
                                    "panel_effective_date": "1/15/2021",
                                    "community_name": "HERNANDO COUNTY",
                                    "community_id": "",
                                    "zone_found": "yes",
                                    "no_digital_data": "no",
                                    "map_unreadable": "no",
                                },
                            }
                        }
                    ]
                }
            }
        }
    )

    extraction = flood_zone_extractor.extract_flood_zone(b"%PDF-fake-bytes", bedrock_client=client)

    assert extraction.success is False
    assert extraction.error == "invalid_primary_fields"


def test_schema_puts_reasoning_first() -> None:
    # The `reasoning` field is load-bearing, not cosmetic: the model fills
    # schema properties in order, so reasoning-first forces it to locate the
    # property marker and describe the shading BEFORE committing to a
    # classification (evaluated at 58/60 vs 2/4 without it). A refactor that
    # reorders the dict silently degrades accuracy.
    assert next(iter(flood_zone_extractor.SCHEMA["properties"])) == "reasoning"
    assert flood_zone_extractor.SCHEMA["required"][0] == "reasoning"


def test_extract_flood_zone_captures_reasoning_and_bounds_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_render(monkeypatch)
    long_reasoning = "the marker sits on unshaded ground " * 100
    client = _FakeBedrockClient(
        {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "extract_flood_zone",
                                "input": {
                                    "reasoning": long_reasoning,
                                    "flood_zone": "",
                                    "sfha": "no",
                                    "panel_number": "",
                                    "panel_effective_date": "",
                                    "community_name": "",
                                    "community_id": "",
                                    "zone_found": "yes",
                                    "no_digital_data": "no",
                                    "map_unreadable": "no",
                                },
                            }
                        }
                    ]
                }
            }
        }
    )

    extraction = flood_zone_extractor.extract_flood_zone(b"%PDF-fake-bytes", bedrock_client=client)

    assert extraction.success
    assert extraction.reasoning == long_reasoning.strip()
    # to_log_dict is embedded in workflow results and CloudWatch lines; a long
    # deliberation must not bloat them.
    assert len(extraction.to_log_dict()["reasoning"]) <= 2000


def test_extract_flood_zone_requests_token_headroom_for_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    fake_bedrock_extraction_response: dict[str, Any],
) -> None:
    # With reasoning serialized first and the gate fields (zone_found,
    # no_digital_data, map_unreadable) LAST, a max_tokens truncation blanks
    # the gates and silently converts correct classifications into
    # unconfident-errored runs. 512 was observed doing exactly that.
    _stub_render(monkeypatch)
    client = _FakeBedrockClient(fake_bedrock_extraction_response)

    flood_zone_extractor.extract_flood_zone(b"%PDF-fake-bytes", bedrock_client=client)

    assert client.calls[0]["inferenceConfig"]["maxTokens"] >= 2048


class _FakePixmap:
    def __init__(self, size: int) -> None:
        self._size = size

    def tobytes(self, _fmt: str) -> bytes:
        return b"\x89" * self._size


class _FakePage:
    """Pretends PNG size scales with pixel count (~dpi^2), like real renders."""

    def __init__(self, bytes_at_150_dpi: int) -> None:
        self.bytes_at_150_dpi = bytes_at_150_dpi
        self.rendered_dpis: list[int] = []

    def get_pixmap(self, *, dpi: int, alpha: bool) -> _FakePixmap:
        self.rendered_dpis.append(dpi)
        return _FakePixmap(int(self.bytes_at_150_dpi * (dpi / 150) ** 2))


def test_render_page_within_budget_passes_through_fitting_pages() -> None:
    page = _FakePage(bytes_at_150_dpi=3_000_000)

    png = flood_zone_extractor._render_page_png_within_budget(page, page_index=0, dpi=150)

    assert len(png) == 3_000_000
    assert page.rendered_dpis == [150]


def test_render_page_within_budget_downscales_oversized_page() -> None:
    # The live failure shape: 3,954,187 bytes at 150 DPI encodes past Bedrock's
    # 5 MB base64 cap. One proportional re-render must land under budget.
    page = _FakePage(bytes_at_150_dpi=3_954_187)

    png = flood_zone_extractor._render_page_png_within_budget(page, page_index=0, dpi=150)

    assert len(png) <= flood_zone_extractor.MAX_PAGE_PNG_BYTES
    assert len(page.rendered_dpis) == 2
    assert page.rendered_dpis[1] < 150
    assert page.rendered_dpis[1] >= flood_zone_extractor._MIN_RENDER_DPI


def test_render_page_within_budget_raises_when_floor_still_oversized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A page so heavy that even the minimum-DPI render exceeds the cap must
    # fail controlled (render_failed:oversized_page_render downstream) rather
    # than send a request Bedrock is guaranteed to reject.
    page = _FakePage(bytes_at_150_dpi=20_000_000)

    with pytest.raises(ValueError, match="oversized_page_render"):
        flood_zone_extractor._render_page_png_within_budget(page, page_index=0, dpi=150)
