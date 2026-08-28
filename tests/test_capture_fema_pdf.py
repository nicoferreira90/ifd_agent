"""Smoke tests for the FEMA NFHL Playwright capture helpers (no real browser launched)."""

import asyncio
import re
from collections.abc import Iterator
from html import escape
from typing import Any

import pytest
from af.tools import logger as af_logger
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from ifd_agent.playwright_assistant import capturePDF_async
from ifd_agent.playwright_assistant.capturePDF_async import (
    NFHL_VIEWER_URL,
    _build_search_string,
    _capture_nfhl_page_pdf,
    _click_first_visible_selector,
    _dismiss_nfhl_modals,
    _fill_first_visible_selector,
    _navigate_to_nfhl_viewer,
    _search_nfhl_for_address,
    _wait_for_map_images_rendered,
    capture_fema_pdf_async,
)


@pytest.fixture(autouse=True)
def _instant_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the helpers' `asyncio.sleep` calls instant so unit tests stay fast.

    The capture helpers use generous sleeps (e.g. 8s for tile rasterization) to
    let real Esri canvases settle. Those waits add nothing to unit tests that
    drive a `_FakePage`, so we replace them with a no-op coroutine.
    """

    async def _no_sleep(_seconds: float, *_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(capturePDF_async.asyncio, "sleep", _no_sleep)


@pytest.fixture
def loguru_messages() -> Iterator[list[str]]:
    """Capture loguru messages emitted via `af.tools.logger` into a list.

    `caplog` only intercepts stdlib logging; `af.tools.logger` is a loguru
    logger so we register a list sink for the duration of the test.
    """
    messages: list[str] = []
    sink_id = af_logger.add(lambda msg: messages.append(msg.record["message"]), level="DEBUG")
    try:
        yield messages
    finally:
        af_logger.remove(sink_id)


class _FakeInput:
    def __init__(
        self,
        selector: str,
        fills_log: list[tuple[str, str]],
        clicks_log: list[tuple[str, str]],
        evaluates_log: list[tuple[str, str]] | None = None,
        raise_on_evaluate: Exception | None = None,
        attributes: dict[str, str] | None = None,
    ) -> None:
        self.selector = selector
        self.fills_log = fills_log
        self.clicks_log = clicks_log
        self.evaluates_log = evaluates_log if evaluates_log is not None else []
        self.raise_on_evaluate = raise_on_evaluate
        self.attributes = dict(attributes) if attributes else {}

    async def fill(self, value: str) -> None:
        await asyncio.sleep(0)
        self.fills_log.append((self.selector, value))

    async def click(self) -> None:
        await asyncio.sleep(0)
        self.clicks_log.append((self.selector, "click"))

    async def evaluate(self, script: str) -> None:
        await asyncio.sleep(0)
        if self.raise_on_evaluate is not None:
            raise self.raise_on_evaluate
        self.evaluates_log.append((self.selector, script))

    async def get_attribute(self, name: str) -> str | None:
        await asyncio.sleep(0)
        return self.attributes.get(name)


class _FakeKeyboard:
    def __init__(self) -> None:
        self.presses: list[str] = []

    async def press(self, key: str) -> None:
        await asyncio.sleep(0)
        self.presses.append(key)


class _FakePage:
    """A page double that records interactions for assertion in tests."""

    def __init__(self, visible_selectors: set[str]) -> None:
        self.visible_selectors = set(visible_selectors)
        self.fills: list[tuple[str, str]] = []
        self.clicks: list[tuple[str, str]] = []
        self.input_evaluates: list[tuple[str, str]] = []
        self.gotos: list[tuple[str, dict[str, Any]]] = []
        self.load_states: list[str] = []
        self.media_emulations: list[str] = []
        self.viewport_sizes: list[dict[str, int]] = []
        self.pdf_calls: list[dict[str, Any]] = []
        self.evaluate_calls: list[tuple[str, Any]] = []
        self.evaluate_returns: list[Any] = []
        self.pdf_return_value: bytes = b"%PDF-1.7\n%fake pdf\n"
        self.keyboard = _FakeKeyboard()
        # When set, the next `_FakeInput.evaluate()` raises this exception
        # instead of recording — used to simulate a JS-click failure.
        self.input_evaluate_error: Exception | None = None
        # Per-selector attribute maps so helpers can call
        # `element.get_attribute('href')`. Keyed by the matched selector
        # candidate, not the joined selector string passed in.
        self.element_attributes: dict[str, dict[str, str]] = {}
        # Patterns that wait_for_url should accept without raising. Add a
        # glob like "**/portal/search**" to permit a transition.
        self.allowed_url_patterns: list[str] = []
        self.wait_for_url_calls: list[str] = []
        # wait_for_function (map-image readiness probe) bookkeeping. By default
        # the probe resolves immediately (images ready); set
        # `map_images_never_ready=True` to simulate a render that never settles.
        self.wait_for_function_calls: list[str] = []
        self.map_images_never_ready: bool = False
        # Live page URL — the Esri viewer carries the property's `&extent=` here
        # after the geocoder zoom; the captured PDF footer prints it.
        self.url: str = (
            "https://hazards-fema.maps.arcgis.com/apps/webappviewer/index.html"
            "?id=8b0adb51996444d4879338b5529aa9cd&extent=-80.94,35.39,-80.89,35.41"
        )

    async def wait_for_selector(self, selector: str, **kwargs: object) -> _FakeInput:
        await asyncio.sleep(0)
        # Real Playwright treats `"a, b, c"` as a union — first match wins.
        candidates = [s.strip() for s in selector.split(",")]
        for candidate in candidates:
            if candidate in self.visible_selectors:
                return _FakeInput(
                    candidate,
                    self.fills,
                    self.clicks,
                    evaluates_log=self.input_evaluates,
                    raise_on_evaluate=self.input_evaluate_error,
                    attributes=self.element_attributes.get(candidate),
                )
        # Missing-selector waits raise TimeoutError in real Playwright. Helpers
        # should swallow only timeouts, not generic exceptions.
        raise PlaywrightTimeoutError(f"selector not visible: {selector}")

    async def goto(self, url: str, **kwargs: Any) -> None:
        await asyncio.sleep(0)
        self.gotos.append((url, dict(kwargs)))

    async def wait_for_url(self, pattern: str, **_kwargs: object) -> None:
        await asyncio.sleep(0)
        self.wait_for_url_calls.append(pattern)
        # Accept any pattern by default; tests inject explicit allowed_url_patterns
        # only when they want to assert behavior, otherwise this is a no-op.

    async def wait_for_function(self, expression: str, **_kwargs: object) -> None:
        await asyncio.sleep(0)
        self.wait_for_function_calls.append(expression)
        if self.map_images_never_ready:
            raise PlaywrightTimeoutError("map images not ready")

    async def wait_for_load_state(self, state: str = "load", **_kwargs: object) -> None:
        await asyncio.sleep(0)
        self.load_states.append(state)

    async def set_viewport_size(self, size: dict[str, int]) -> None:
        await asyncio.sleep(0)
        self.viewport_sizes.append(dict(size))

    async def emulate_media(self, media: str = "screen", **_kwargs: object) -> None:
        await asyncio.sleep(0)
        self.media_emulations.append(media)

    async def evaluate(self, script: str, arg: Any = None) -> Any:
        await asyncio.sleep(0)
        self.evaluate_calls.append((script, arg))
        if not self.evaluate_returns:
            return None
        value = self.evaluate_returns.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    async def pdf(self, **kwargs: Any) -> bytes:
        await asyncio.sleep(0)
        self.pdf_calls.append(dict(kwargs))
        return self.pdf_return_value


def test_build_search_string_prefers_formatted_value() -> None:
    address = {
        "street": "1600 Pennsylvania Ave NW",
        "city": "Washington",
        "state": "DC",
        "postal_code": "20500",
        "formatted": "1600 Pennsylvania Ave NW, Washington, DC 20500",
    }
    assert _build_search_string(address) == "1600 Pennsylvania Ave NW, Washington, DC 20500"


def test_build_search_string_concatenates_components_when_no_formatted() -> None:
    address = {
        "street": "1 Apple Park Way",
        "unit": "Bldg 4",
        "city": "Cupertino",
        "state": "CA",
        "postal_code": "95014",
    }
    assert _build_search_string(address) == "1 Apple Park Way Bldg 4, Cupertino, CA, 95014"


def test_build_search_string_handles_missing_fields() -> None:
    address = {"street": "", "city": "Cupertino", "state": "CA", "postal_code": ""}
    assert _build_search_string(address) == "Cupertino, CA"


def test_build_search_string_strips_zip_plus_four_from_components() -> None:
    # Encompass loans routinely carry ZIP+4 (e.g. 70122-1937). Esri's NFHL
    # geocoder rejects the +4 form — it returns zero autocomplete results
    # when the search string includes the +4 extension, which breaks the
    # menuitem-click step and means the map never zooms to the property.
    address = {
        "street": "1478 Riviera Ave",
        "city": "New Orleans",
        "state": "LA",
        "postal_code": "70122-1937",
    }
    result = _build_search_string(address)
    assert result == "1478 Riviera Ave, New Orleans, LA, 70122"
    assert "-1937" not in result


def test_build_search_string_strips_zip_plus_four_from_formatted_field() -> None:
    # When `formatted` is set we use it directly — strip ZIP+4 there too,
    # otherwise the geocoder still sees the +4 form.
    address = {
        "street": "1478 Riviera Ave",
        "city": "New Orleans",
        "state": "LA",
        "postal_code": "70122-1937",
        "formatted": "1478 Riviera Ave, New Orleans, LA 70122-1937",
    }
    result = _build_search_string(address)
    assert result == "1478 Riviera Ave, New Orleans, LA 70122"


def test_fill_first_visible_selector_returns_true_on_match() -> None:
    page = _FakePage({"input#searchstring"})

    filled = asyncio.run(
        _fill_first_visible_selector(
            page, ["input#searchstring", "input[type='text']"], "value", "FEMA"
        )
    )

    assert filled
    assert page.fills == [("input#searchstring", "value")]


def test_fill_first_visible_selector_returns_false_when_no_selector_matches() -> None:
    page = _FakePage(set())

    filled = asyncio.run(_fill_first_visible_selector(page, ["input#absent"], "value", "FEMA"))

    assert filled is False


def test_click_first_visible_selector_clicks_first_match() -> None:
    page = _FakePage({"button[type='submit']"})

    clicked = asyncio.run(
        _click_first_visible_selector(page, ["button#absent", "button[type='submit']"], "Submit")
    )

    assert clicked
    assert page.clicks == [("button[type='submit']", "click")]
    assert page.fills == []


_SAMPLE_ADDRESS = "10905 Claywood Dr, Austin, TX, 78753"


def test_navigate_to_nfhl_viewer_goes_to_known_url() -> None:
    page = _FakePage(set())

    asyncio.run(_navigate_to_nfhl_viewer(page))

    assert page.gotos and page.gotos[0][0] == NFHL_VIEWER_URL


def test_dismiss_nfhl_modals_dismisses_all_until_idle(
    loguru_messages: list[str],
) -> None:
    page = _FakePage(set())
    # Three iterations: 2 modals on first pass (welcome + layer error),
    # 1 modal on second pass (revealed by closing first), 0 on third.
    page.evaluate_returns = [2, 1, 0]

    total = asyncio.run(_dismiss_nfhl_modals(page))

    assert total == 3
    # The first two evaluate calls happen *because* the previous one returned >0.
    assert len(page.evaluate_calls) == 3
    joined = "\n".join(loguru_messages)
    assert "ifd_nfhl_modals_dismissed attempt=1 count=2" in joined
    assert "ifd_nfhl_modals_dismissed attempt=2 count=1" in joined
    assert "ifd_nfhl_modals_total_dismissed count=3" in joined


def test_dismiss_nfhl_modals_returns_zero_when_no_modals(
    loguru_messages: list[str],
) -> None:
    page = _FakePage(set())
    page.evaluate_returns = [0]

    total = asyncio.run(_dismiss_nfhl_modals(page))

    assert total == 0
    assert any("ifd_nfhl_no_modals_present" in msg for msg in loguru_messages)


def test_dismiss_nfhl_modals_stops_after_max_iterations() -> None:
    # Pathological case: a modal keeps re-appearing. We should cap iterations.
    page = _FakePage(set())
    page.evaluate_returns = [1, 1, 1, 1, 1, 1, 1, 1]  # plenty more than max

    total = asyncio.run(_dismiss_nfhl_modals(page, max_iterations=3))

    # Exactly max_iterations evaluate calls — we don't loop forever.
    assert total == 3
    assert len(page.evaluate_calls) == 3


def test_dismiss_nfhl_modals_returns_zero_when_evaluate_raises(
    loguru_messages: list[str],
) -> None:
    page = _FakePage(set())
    page.evaluate_returns = [RuntimeError("eval boom")]

    total = asyncio.run(_dismiss_nfhl_modals(page))

    # Non-fatal: capture flow should continue even if the JS pass errors.
    assert total == 0
    assert any(
        "ifd_nfhl_modal_dismiss_eval_error" in msg and "eval boom" in msg for msg in loguru_messages
    )


def test_search_nfhl_fills_search_box_and_submits() -> None:
    page = _FakePage({"input.esriGeocoder", ".esriGeocoderSearch", ".esriPopup .title"})

    asyncio.run(_search_nfhl_for_address(page, _SAMPLE_ADDRESS))

    assert ("input.esriGeocoder", _SAMPLE_ADDRESS) in page.fills
    assert (".esriGeocoderSearch", "click") in page.clicks
    assert page.keyboard.presses == []


def test_search_nfhl_falls_back_to_enter_when_no_submit_button() -> None:
    # Geocoder input visible but none of the submit candidates.
    page = _FakePage({"input.esriGeocoder", ".esriPopup .title"})

    asyncio.run(_search_nfhl_for_address(page, _SAMPLE_ADDRESS))

    assert ("input.esriGeocoder", _SAMPLE_ADDRESS) in page.fills
    assert page.keyboard.presses == ["Enter"]


def test_search_nfhl_raises_when_geocoder_input_missing() -> None:
    page = _FakePage(set())

    with pytest.raises(RuntimeError, match="Could not locate NFHL search input"):
        asyncio.run(_search_nfhl_for_address(page, _SAMPLE_ADDRESS))


def test_search_nfhl_logs_warning_when_popup_never_appears(
    loguru_messages: list[str],
) -> None:
    # Geocoder and submit exist; popup selectors are not visible.
    page = _FakePage({"input.esriGeocoder", ".esriGeocoderSearch"})

    asyncio.run(_search_nfhl_for_address(page, _SAMPLE_ADDRESS))

    assert any("ifd_nfhl_map_ready_no_popup" in msg for msg in loguru_messages)


def test_capture_nfhl_page_pdf_returns_bytes_and_size(loguru_messages: list[str]) -> None:
    page = _FakePage(set())
    # The msc.fema.gov hand-off URL already carries the property's &extent=.
    map_url = (
        "https://hazards-fema.maps.arcgis.com/apps/webappviewer/index.html"
        "?id=8b0adb51996444d4879338b5529aa9cd&extent=-97.77,30.32,-97.61,30.43"
    )

    pdf_bytes, pdf_size = asyncio.run(_capture_nfhl_page_pdf(page, map_url))

    assert pdf_bytes == page.pdf_return_value
    assert pdf_size == len(page.pdf_return_value)
    assert page.media_emulations == ["screen"]
    assert page.pdf_calls and page.pdf_calls[0]["format"] == "Letter"
    assert page.pdf_calls[0]["landscape"] is True
    # Header reproduces Chrome's default print header: native date token (left)
    # + the fixed page title (right), matching the manual artifact.
    header = page.pdf_calls[0]["header_template"]
    assert 'class="date"' in header
    assert escape(capturePDF_async.NFHL_VIEWER_PAGE_TITLE) in header
    # Footer carries the hand-off URL passed in (with &extent=) on the left and
    # the native page-number token on the right — not a timestamp, not page.url.
    # The URL is HTML-escaped in the template (& -> &amp;).
    footer = page.pdf_calls[0]["footer_template"]
    assert escape(map_url) in footer
    assert "extent=-97.77,30.32,-97.61,30.43" in footer
    assert 'class="pageNumber"' in footer
    assert 'class="totalPages"' in footer
    assert not re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", footer)
    assert any("ifd_pdf_captured" in msg for msg in loguru_messages)


def test_wait_for_map_images_rendered_returns_true_when_ready(
    loguru_messages: list[str],
) -> None:
    page = _FakePage(set())

    ready = asyncio.run(_wait_for_map_images_rendered(page))

    assert ready is True
    # The readiness probe ran exactly once against the page.
    assert len(page.wait_for_function_calls) == 1
    assert "naturalWidth" in page.wait_for_function_calls[0]
    assert any("ifd_nfhl_map_images_ready" in msg for msg in loguru_messages)


def test_wait_for_map_images_rendered_returns_false_on_timeout(
    loguru_messages: list[str],
) -> None:
    page = _FakePage(set())
    page.map_images_never_ready = True

    ready = asyncio.run(_wait_for_map_images_rendered(page))

    # Best-effort: a never-settling render returns False (caller proceeds to
    # capture; the vision map_unreadable check is the backstop) and never raises.
    assert ready is False
    assert any("ifd_nfhl_map_images_wait_timeout" in msg for msg in loguru_messages)


def test_search_nfhl_proceeds_when_map_images_never_render(
    loguru_messages: list[str],
) -> None:
    # Even if the overlay images never finish, the search/capture flow must not
    # crash — it logs and proceeds so the downstream vision check can flag the
    # corrupted map.
    page = _FakePage({"input.esriGeocoder", ".esriGeocoderSearch", '[role="menuitem"]'})
    page.map_images_never_ready = True

    asyncio.run(_search_nfhl_for_address(page, _SAMPLE_ADDRESS))

    assert page.wait_for_function_calls  # the readiness probe was attempted
    assert any("images_ready=False" in msg for msg in loguru_messages)


class _FakeBrowser:
    def __init__(self, page: _FakePage) -> None:
        self._page = page
        self.closed = False
        self.launch_args: list[Any] = []

    async def new_page(self) -> _FakePage:
        await asyncio.sleep(0)
        return self._page

    async def close(self) -> None:
        await asyncio.sleep(0)
        self.closed = True


class _FakeChromium:
    def __init__(self, browser: _FakeBrowser) -> None:
        self._browser = browser

    async def launch(self, **kwargs: Any) -> _FakeBrowser:
        await asyncio.sleep(0)
        self._browser.launch_args.append(kwargs)
        return self._browser


class _FakePlaywrightCM:
    def __init__(self, page: _FakePage) -> None:
        self._browser = _FakeBrowser(page)
        self.chromium = _FakeChromium(self._browser)
        self.browser = self._browser

    async def __aenter__(self) -> "_FakePlaywrightCM":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


def test_capture_fema_pdf_async_drives_msc_fema_then_nfhl_flow(
    monkeypatch: pytest.MonkeyPatch,
    loguru_messages: list[str],
) -> None:
    extracted_viewer_url = (
        "https://hazards-fema.maps.arcgis.com/apps/webappviewer/index.html"
        "?id=8b0adb51996444d4879338b5529aa9cd&extent=-97.77,30.32,-97.61,30.43"
    )
    page = _FakePage(
        {
            # msc.fema.gov stage
            "input#txtAddressSearch",
            "input#addressLocate",
            'a:has-text("Go To NFHL Viewer")',
            # NFHL viewer stage
            "input.esriGeocoder",
            ".esriGeocoderSearch",
            ".esriPopup .title",
            '[role="menuitem"]',
        }
    )
    # Stub the href the msc.fema.gov result page returns for the viewer link.
    page.element_attributes['a:has-text("Go To NFHL Viewer")'] = {
        "href": extracted_viewer_url,
    }
    fake_cm = _FakePlaywrightCM(page)
    monkeypatch.setattr(capturePDF_async, "async_playwright", lambda: fake_cm)

    pdf_bytes, pdf_size = asyncio.run(capture_fema_pdf_async({"formatted": _SAMPLE_ADDRESS}))

    # Two gotos, in order: msc.fema.gov first, then the property-specific
    # NFHL viewer URL extracted from that page.
    goto_urls = [url for url, _kwargs in page.gotos]
    assert goto_urls == [capturePDF_async.FEMA_PORTAL_URL, extracted_viewer_url]

    # msc.fema.gov search was filled + submit button clicked (no Enter
    # fallback because the submit selector matched).
    assert ("input#txtAddressSearch", _SAMPLE_ADDRESS) in page.fills
    assert ("input#addressLocate", "click") in page.clicks
    assert "Enter" not in page.keyboard.presses

    # NFHL viewer search was filled, submit was clicked, autocomplete menuitem
    # was clicked to commit the geocode.
    assert ("input.esriGeocoder", _SAMPLE_ADDRESS) in page.fills
    assert (".esriGeocoderSearch", "click") in page.clicks
    assert ('[role="menuitem"]', "click") in page.clicks

    assert page.pdf_calls and page.pdf_calls[0]["landscape"] is True

    # Footer prints the msc.fema.gov hand-off URL (which already carries the
    # property's &extent=), NOT page.url. The viewer strips the extent off
    # window.location during boot, so page.url's extent must not leak through.
    footer = page.pdf_calls[0]["footer_template"]
    assert escape(extracted_viewer_url) in footer
    assert "extent=-97.77,30.32,-97.61,30.43" in footer
    assert "-80.94,35.39,-80.89,35.41" not in footer  # page.url's stale extent

    # Return matches the stubbed PDF.
    assert pdf_bytes == page.pdf_return_value
    assert pdf_size == len(page.pdf_return_value)

    # Browser was closed cleanly.
    assert fake_cm.browser.closed is True

    # Key log markers across both stages.
    joined = "\n".join(loguru_messages)
    assert "ifd_capture_start" in joined
    assert "ifd_msc_fema_navigate" in joined
    assert "ifd_msc_fema_nfhl_url_extracted" in joined
    assert "ifd_nfhl_navigate" in joined
    assert "ifd_nfhl_search_submitted" in joined
    assert "ifd_nfhl_geocoder_result_selected" in joined
    assert "ifd_pdf_captured" in joined


def test_capture_fema_pdf_async_falls_back_to_canonical_url_when_msc_fema_fails(
    monkeypatch: pytest.MonkeyPatch,
    loguru_messages: list[str],
) -> None:
    # msc.fema.gov search input isn't available → URL extraction fails →
    # orchestrator falls back to NFHL_VIEWER_URL. Capture should still
    # complete; it just loses the property-specific extent.
    page = _FakePage(
        {
            "input.esriGeocoder",
            ".esriGeocoderSearch",
            ".esriPopup .title",
            '[role="menuitem"]',
        }
    )
    fake_cm = _FakePlaywrightCM(page)
    monkeypatch.setattr(capturePDF_async, "async_playwright", lambda: fake_cm)

    asyncio.run(capture_fema_pdf_async({"formatted": _SAMPLE_ADDRESS}))

    goto_urls = [url for url, _kwargs in page.gotos]
    # First goto is msc.fema.gov (search input missing → bail), second is
    # the canonical NFHL viewer URL we fell back to.
    assert goto_urls == [capturePDF_async.FEMA_PORTAL_URL, NFHL_VIEWER_URL]
    joined = "\n".join(loguru_messages)
    assert "ifd_msc_fema_search_input_not_found" in joined
    assert "ifd_msc_fema_fallback_to_canonical_url" in joined


def test_capture_fema_pdf_async_rejects_empty_address() -> None:
    with pytest.raises(ValueError, match="no usable street/city/state/postal_code"):
        asyncio.run(capture_fema_pdf_async({}))


class _ErrorOnWaitPage(_FakePage):
    """Page double whose `wait_for_selector` always raises a non-timeout error.

    Used to confirm the selector helpers do NOT swallow unexpected exceptions
    (only `PlaywrightTimeoutError` should fall through to the next candidate).
    """

    async def wait_for_selector(self, selector: str, **kwargs: object) -> Any:
        await asyncio.sleep(0)
        raise RuntimeError("simulated browser-target-closed")


def test_fill_first_visible_selector_propagates_non_timeout_errors() -> None:
    page = _ErrorOnWaitPage(set())

    with pytest.raises(RuntimeError, match="simulated browser-target-closed"):
        asyncio.run(_fill_first_visible_selector(page, ["input#anything"], "value", "label"))


# Note: the modal dismissal helper no longer uses wait_for_selector — it
# runs a single JS pass via page.evaluate() and catches all errors as
# non-fatal (see test_dismiss_nfhl_modals_returns_zero_when_evaluate_raises).
