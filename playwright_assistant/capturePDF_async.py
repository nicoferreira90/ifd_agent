"""FEMA flood map PDF capture, two-stage flow.

Stage 1 — msc.fema.gov (the FEMA Map Service Center portal): we search the
subject property here for two reasons. The page produces a property-specific
"Go To NFHL Viewer" link whose href encodes the lat/long extent, AND the
visit establishes `*.fema.gov` session cookies that the NFHL viewer's
downstream layer services check before serving FIRM Panels / NFHL data.
Skipping this stage results in `credential is null` errors in the browser
console and the flood-zone overlay never renders in the captured page.

Stage 2 — NFHL Web AppViewer (Esri-hosted at hazards-fema.maps.arcgis.com):
we navigate to the URL extracted in stage 1, dismiss the welcome splash and
any layer-error popups, search the address again in the viewer's geocoder,
click the autocomplete result to commit the geocode (this is what triggers
the actual zoom-and-load behavior), wait for tiles to render, and capture
the page as a PDF.

The captured page is saved as `FLOODSEARCH.pdf` for upload to the Encompass
eFolder. The downstream vision extractor reads the PDF to extract the flood
zone code, so this module focuses on producing a clean rendered map with
the FIRM Panels overlay visible — not on DOM scraping (the NFHL viewer is
a JS-heavy Esri app and its selectors churn).
"""

import asyncio
import re
from html import escape
from typing import Any

from af.tools import logger
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

BROWSER_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-setuid-sandbox",
    "--disable-background-networking",
    "--disable-client-side-phishing-detection",
    "--disable-component-update",
    "--disable-domain-reliability",
    "--disable-features=AudioServiceOutOfProcess",
    "--disable-hang-monitor",
    "--disable-ipc-flooding-protection",
    "--disable-popup-blocking",
    "--disable-prompt-on-repost",
    "--disable-renderer-backgrounding",
    "--disable-sync",
    "--force-color-profile=srgb",
    "--metrics-recording-only",
    "--mute-audio",
    "--no-pings",
    # NOTE: previously this list also carried `--disable-gpu`, `--single-process`,
    # `--no-zygote`, and `--disable-accelerated-2d-canvas`. Those four together
    # prevented Chromium from spawning the GPU process used by SwiftShader for
    # software WebGL, which the NFHL viewer's vector tile layers need to render
    # the flood-zone overlays. Removed once we confirmed a coworker's reference
    # script (which uses minimal args) loads the layers correctly.
]

# FEMA Map Service Center portal. We hit this *first* to search the address.
# The resulting page exposes a "Go To NFHL Viewer" link whose href includes
# the property's lat/long extent — but more importantly, the visit establishes
# `*.fema.gov` session cookies that the NFHL viewer's downstream layer
# services check before serving FIRM Panels / NFHL data. Skipping this stage
# results in `credential is null` errors and the flood overlay never renders.
FEMA_PORTAL_URL = "https://msc.fema.gov/portal/home"

# Candidate CSS selectors for the msc.fema.gov search input. The portal
# occasionally reskins; we try a small set in order. Confirmed primary ID
# (May 2026): `txtAddressSearch`.
FEMA_PORTAL_SEARCH_SELECTORS = [
    "input#txtAddressSearch",
    'input[placeholder*="address" i]',
    'input[type="search"]',
    'input[type="text"]',
]

# Candidate submit-button selectors on the msc.fema.gov home page. Pressing
# Enter on the search input occasionally fails to trigger navigation in
# headless Chromium; clicking the actual button is more reliable. We fall
# back to Enter if no submit selector matches.
FEMA_PORTAL_SUBMIT_SELECTORS = [
    "input#addressLocate",
    'input[type="button"][value="Search"]',
    "button#btnAddressSearch",
    'button[type="submit"]',
    'input[type="submit"]',
]

# FEMA's National Flood Hazard Layer Web AppViewer. Used as a fallback if the
# msc.fema.gov stage can't extract a property-specific viewer URL. The
# property-specific URL (from the "Go To NFHL Viewer" link) is preferred
# because it includes the extent params that pre-position the viewer.
NFHL_VIEWER_URL = (
    "https://hazards-fema.maps.arcgis.com/apps/webappviewer/"
    "index.html?id=8b0adb51996444d4879338b5529aa9cd"
)

# Page title printed in the captured PDF header (right side), matching the
# manual Chrome-print artifact. Hard-coded rather than read from the native
# `class="title"` token / `document.title` so it matches even if headless
# Chromium reports a different/empty title.
NFHL_VIEWER_PAGE_TITLE = "FEMA's National Flood Hazard Layer (NFHL) Viewer"

# CSS selector for the "Go To NFHL Viewer" link on the msc.fema.gov results
# page. Playwright's `:has-text` extension matches against textContent.
NFHL_VIEWER_LINK_SELECTORS = [
    'a:has-text("Go To NFHL Viewer")',
    'a[href*="hazards-fema.maps.arcgis.com"]',
]

# JS pass that finds any visible modal dialog (Welcome splash, layer-error
# popup, widget panel, etc.) and dispatches a click on its OK/Close button
# via the element's onclick handler. This bypasses Esri's `jimu-overlay`,
# which intercepts pointer events in headless Chromium and makes native
# Playwright clicks time out. Returns the number of dialogs dismissed in
# this pass so the caller can loop until idle.
_NFHL_DISMISS_DIALOGS_JS = """
() => {
    function isVisible(el) {
        if (!el || !el.getBoundingClientRect) return false;
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') return false;
        if (parseFloat(style.opacity || '1') === 0) return false;
        return true;
    }

    const dialogSelectors = [
        '[role="dialog"]',
        // The NFHL Welcome splash container in production renders with
        // aria-label="Splash" (confirmed via coworker's reference script's
        // `get_by_label("Splash", exact=True)` working in the same env).
        // None of the class-based selectors below match it, so we have to
        // hit it via aria-label.
        '[aria-label="Splash"]',
        '[aria-label*="Welcome" i]',
        '.jimu-splash',
        '.jimu-splash-controller',
        '.jimu-confirm-dialog',
        '.jimu-popup',
        '.dijitDialog',
        '.dijitDialogPaneContent',
        '.jimu-on-screen-widget-panel',
    ];
    const seen = new Set();
    let dismissed = 0;

    for (const sel of dialogSelectors) {
        const dialogs = document.querySelectorAll(sel);
        for (const dialog of dialogs) {
            if (seen.has(dialog) || !isVisible(dialog)) continue;
            seen.add(dialog);

            // Look for an OK/Close-style action button inside this dialog.
            const candidates = dialog.querySelectorAll(
                '[role="button"], button, .jimu-btn, .dijitButton, .close-btn'
            );
            for (const btn of candidates) {
                if (!isVisible(btn)) continue;
                const text = (btn.textContent || '').trim().toUpperCase();
                const aria = (btn.getAttribute('aria-label') || '').toUpperCase();
                if (
                    text === 'OK' || text === 'OKAY' || text === 'CLOSE' ||
                    aria.includes('CLOSE')
                ) {
                    btn.click();
                    dismissed += 1;
                    break;  // one dismissal per dialog
                }
            }
        }
    }
    return dismissed;
}
"""

# Esri geocoder search input inside the webappviewer.
NFHL_SEARCH_SELECTORS = [
    "input.esriGeocoder",
    ".esriGeocoder input",
    "input[placeholder*='Find' i]",
    "input[placeholder*='address' i]",
    "input[type='text']",
]

# Esri geocoder submit button (magnifier icon).
NFHL_SUBMIT_SELECTORS = [
    ".esriGeocoderSearch",
    ".esriGeocoder .searchSubmit",
    "[title*='Search' i]",
]

# DOM marker that appears once the search has resolved and the map has panned
# to the property. The NFHL search-result popup contains the searched address.
NFHL_RESULT_POPUP_SELECTORS = [
    ".esriPopup .title",
    ".esriPopupWrapper",
    "[class*='popup'][class*='wrapper']",
]

# Readiness gate evaluated in the page before we rasterize the PDF. The NFHL
# flood overlay is an ArcGIS dynamic-map-service export PNG rendered as <img>
# element(s) layered ON TOP of the basemap tiles. On this slow site those
# overlay images sometimes paint late or fail to decode, which is what produces
# the corrupted / half-rendered captures the vision extractor flags as
# map_unreadable. This predicate returns true only once the overlay image(s)
# are present, laid out, and fully decoded (complete + non-zero natural size),
# so we capture a fully-painted map. Patterns target the FEMA NFHL FIRMette
# export images (by element id or MapServer export src), with a broader NFHL
# export fallback in case the service name varies.
_MAP_IMAGES_RENDERED_JS = r"""
() => {
    const visible = (n) => Boolean(n.offsetWidth || n.offsetHeight || n.getClientRects().length);
    const imgs = Array.from(document.querySelectorAll('img')).filter((img) => {
        const id = img.id || '';
        const src = img.currentSrc || img.src || '';
        return /^map_NFHLREST_FIRMette/.test(id) ||
            /NFHLREST_FIRMette\/MapServer\/export/i.test(src) ||
            /NFHL.*\/MapServer\/export/i.test(src);
    });
    const vis = imgs.filter(visible);
    return vis.length > 0 && vis.every((i) => i.complete && i.naturalWidth > 0 && i.naturalHeight > 0);
}
"""
_MAP_IMAGE_READY_TIMEOUT_MS = 45_000


async def _wait_for_map_images_rendered(page: Any) -> bool:
    """Best-effort wait until the NFHL flood-overlay image(s) finish rendering.

    Returns True once the overlay ``<img>`` element(s) are visible and fully
    decoded, or False if they don't settle within the timeout — in which case we
    still proceed to capture, because the downstream vision ``map_unreadable``
    check is the backstop for a render that never completes. Never raises: a
    slow or garbled render must not crash the capture flow.
    """
    try:
        await page.wait_for_function(_MAP_IMAGES_RENDERED_JS, timeout=_MAP_IMAGE_READY_TIMEOUT_MS)
        logger.info("ifd_nfhl_map_images_ready")
        return True
    except PlaywrightTimeoutError:
        logger.warning("ifd_nfhl_map_images_wait_timeout proceeding_with_capture=true")
        return False
    except Exception as exc:  # noqa: BLE001 - readiness probe must not crash capture
        logger.warning(
            f"ifd_nfhl_map_images_wait_error error_type={type(exc).__name__} "
            f"error={exc} proceeding_with_capture=true"
        )
        return False


async def _fill_first_visible_selector(
    page: Any,
    selectors: list[str],
    value: str,
    label: str,
    wait_timeout_ms: int = 3000,
) -> bool:
    """Fill the first visible selector that exists on the current page.

    Only Playwright timeout errors (selector not present within `wait_timeout_ms`)
    are swallowed to fall through to the next candidate. Other exceptions —
    e.g. page closed, malformed selector — are propagated.
    """
    for selector in selectors:
        try:
            input_field = await page.wait_for_selector(
                selector, timeout=wait_timeout_ms, state="visible"
            )
        except PlaywrightTimeoutError:
            continue
        if input_field:
            await input_field.fill(value)
            logger.info(f"{label} entered using selector: {selector}")
            return True
    return False


async def _click_first_visible_selector(
    page: Any,
    selectors: list[str],
    label: str,
    wait_timeout_ms: int = 3000,
) -> bool:
    """Click the first visible selector that exists on the current page.

    Only Playwright timeout errors (selector not present within `wait_timeout_ms`)
    are swallowed to fall through to the next candidate. Other exceptions are
    propagated.
    """
    for selector in selectors:
        try:
            element = await page.wait_for_selector(
                selector, timeout=wait_timeout_ms, state="visible"
            )
        except PlaywrightTimeoutError:
            continue
        if element:
            await element.click()
            logger.info(f"{label} clicked using selector: {selector}")
            return True
    return False


_ZIP_PLUS_FOUR_RE = re.compile(r"(\b\d{5})-\d{4}\b")


def _strip_zip_plus_four(text: str) -> str:
    """Truncate ZIP+4 (e.g. ``70122-1937``) to the 5-digit ZIP (``70122``).

    Esri's NFHL geocoder doesn't recognize ZIP+4 — it returns zero autocomplete
    results when the search string includes the +4 extension. Encompass loan
    records routinely carry the +4 form for properties USPS has on file, so
    we strip it before handing the address to the geocoder. The 5-digit form
    geocodes reliably.
    """
    return _ZIP_PLUS_FOUR_RE.sub(r"\1", text)


def _build_search_string(address: dict[str, str]) -> str:
    """Build the single-line search string FEMA expects.

    Format mirrors what a user would type into the portal search bar:
    ``"<street> <unit>, <city>, <state>, <postal>"``. ZIP+4 forms are
    normalized to 5-digit ZIPs because Esri's NFHL geocoder rejects +4.
    """
    street = (address.get("street") or "").strip()
    unit = (address.get("unit") or "").strip()
    city = (address.get("city") or "").strip()
    state = (address.get("state") or "").strip()
    postal_code = (address.get("postal_code") or "").strip()

    if address.get("formatted"):
        return _strip_zip_plus_four(str(address["formatted"]).strip())

    postal_code = _strip_zip_plus_four(postal_code)

    street_line = f"{street} {unit}".strip() if unit else street
    locality = ", ".join(part for part in [city, state] if part)
    parts = [p for p in [street_line, locality, postal_code] if p]
    return ", ".join(parts)


async def _get_nfhl_viewer_url_from_msc_fema(page: Any, search_string: str) -> str | None:
    """Hit msc.fema.gov, search the address, and return the NFHL viewer URL.

    Why this stage exists: navigating directly to the NFHL viewer (hardcoded
    URL) leaves the page with no `*.fema.gov` session cookies. The viewer's
    layer services then refuse to issue an Esri token (`credential is null`
    in the browser console) and the FIRM Panels / NFHL flood overlay never
    renders. Going through msc.fema.gov first establishes the session that
    the downstream layer services check. Confirmed working in a coworker's
    reference script.

    Returns the href of the "Go To NFHL Viewer" link on the results page,
    which includes property-specific extent params, or `None` if extraction
    fails. The caller can fall back to `NFHL_VIEWER_URL` in that case.
    """
    logger.info(f"ifd_msc_fema_navigate url={FEMA_PORTAL_URL}")
    await page.goto(FEMA_PORTAL_URL, wait_until="domcontentloaded", timeout=60000)

    filled = await _fill_first_visible_selector(
        page,
        FEMA_PORTAL_SEARCH_SELECTORS,
        search_string,
        "MSC FEMA address",
        wait_timeout_ms=10000,
    )
    if not filled:
        logger.warning("ifd_msc_fema_search_input_not_found")
        return None

    # Try clicking the submit button before falling back to Enter — production
    # observed Enter not triggering navigation in headless Chromium, with
    # `ifd_msc_fema_results_url_not_reached` firing on every run.
    submitted = await _click_first_visible_selector(
        page, FEMA_PORTAL_SUBMIT_SELECTORS, "MSC FEMA Submit", wait_timeout_ms=3000
    )
    if not submitted:
        logger.warning("ifd_msc_fema_submit_button_not_found falling_back=enter")
        await page.keyboard.press("Enter")
    logger.info(f"ifd_msc_fema_search_submitted address='{search_string}'")

    try:
        await page.wait_for_url("**/portal/search**", timeout=30000)
    except PlaywrightTimeoutError:
        # Log the URL we *did* end up on so we can see whether navigation
        # happened to a different URL pattern, or never happened at all.
        current_url = getattr(page, "url", "<unknown>")
        logger.warning(f"ifd_msc_fema_results_url_not_reached current_url={current_url}")
        return None

    try:
        await page.wait_for_load_state("networkidle", timeout=20000)
    except PlaywrightTimeoutError:
        # Results page is JS-heavy; networkidle sometimes never settles.
        # Continue anyway — the link is usually present in the initial DOM.
        pass

    joined = ", ".join(NFHL_VIEWER_LINK_SELECTORS)
    try:
        link = await page.wait_for_selector(joined, timeout=15000, state="visible")
    except PlaywrightTimeoutError:
        logger.warning("ifd_msc_fema_nfhl_link_not_found")
        return None

    if not link:
        return None

    href = await link.get_attribute("href")
    if not href:
        logger.warning("ifd_msc_fema_nfhl_link_no_href")
        return None

    href_str = str(href)
    logger.info(f"ifd_msc_fema_nfhl_url_extracted url={href_str}")
    return href_str


async def _navigate_to_nfhl_viewer(page: Any, viewer_url: str = NFHL_VIEWER_URL) -> None:
    """Open the FEMA NFHL Web AppViewer and let the Esri JS API boot.

    When `viewer_url` is the property-specific URL extracted from msc.fema.gov
    (preferred), the viewer pre-positions on the property AND the navigation
    carries the fema.gov session cookies that the layer services need.
    """
    logger.info(f"ifd_nfhl_navigate url={viewer_url}")
    await page.goto(viewer_url, wait_until="domcontentloaded", timeout=60000)
    try:
        await page.wait_for_load_state("networkidle", timeout=20000)
    except PlaywrightTimeoutError:
        # The viewer never reaches networkidle in some sessions because of
        # background tile re-fetches. Don't block on it.
        pass
    # Generous settle so widgets, popups, and tile init all run before we
    # start interacting. Coworker's reference uses 10s here.
    await asyncio.sleep(10)


async def _dismiss_nfhl_modals(page: Any, max_iterations: int = 5) -> int:
    """Dismiss any visible NFHL viewer modal dialogs (Welcome splash, layer
    errors, widget panels) by JS-clicking their OK/Close action button.

    The viewer opens a "Welcome to the NFHL Viewer" splash with an OK button
    at the bottom, often together with a "...cannot be added to the map"
    layer-error popup also dismissed via OK. Closing one can reveal another,
    so we loop up to `max_iterations` times until no more visible dialogs are
    found.

    All clicks go through `el.click()` in JS rather than Playwright's
    `ElementHandle.click()` — Esri's `jimu-overlay` div intercepts pointer
    events in headless Chromium and makes native clicks time out.

    Returns the total number of dialogs dismissed across all iterations.
    Failures are logged but never raised: capture continues either way.
    """
    total_dismissed = 0
    for attempt in range(max_iterations):
        try:
            count = await page.evaluate(_NFHL_DISMISS_DIALOGS_JS)
        except Exception as exc:
            logger.warning(f"ifd_nfhl_modal_dismiss_eval_error error={exc}")
            break

        if not isinstance(count, int) or count <= 0:
            break

        total_dismissed += count
        logger.info(f"ifd_nfhl_modals_dismissed attempt={attempt + 1} count={count}")
        # Give the next modal (if any) a moment to render before re-scanning.
        await asyncio.sleep(1)

    if total_dismissed == 0:
        logger.info("ifd_nfhl_no_modals_present")
    else:
        logger.info(f"ifd_nfhl_modals_total_dismissed count={total_dismissed}")
    return total_dismissed


async def _search_nfhl_for_address(page: Any, search_string: str) -> None:
    """Fill the NFHL viewer's geocoder, submit, select the autocomplete result,
    and wait for the map to settle.

    The autocomplete-menuitem click (not just pressing Enter / clicking
    Search) is what triggers the geocoder's "zoom to + place pin + load
    overlays" flow. Pressing Enter on the geocoder input alone often submits
    without committing a selection, which is why earlier captures had no
    flood-zone overlay even when the viewer was functional.

    Raises `RuntimeError` if the geocoder input cannot be located.
    """
    filled = await _fill_first_visible_selector(
        page,
        NFHL_SEARCH_SELECTORS,
        search_string,
        "NFHL address",
        wait_timeout_ms=8000,
    )
    if not filled:
        raise RuntimeError("Could not locate NFHL search input on viewer page")

    submitted = await _click_first_visible_selector(
        page, NFHL_SUBMIT_SELECTORS, "NFHL Submit", wait_timeout_ms=3000
    )
    if not submitted:
        logger.warning("ifd_nfhl_submit_button_not_found falling_back=enter")
        await page.keyboard.press("Enter")
    logger.info(f"ifd_nfhl_search_submitted address='{search_string}'")

    # Wait for the autocomplete dropdown and click the first result. This is
    # what actually commits the geocode selection and triggers the map to
    # zoom + drop the pin + load the FIRM Panels overlay for the area.
    try:
        menuitem = await page.wait_for_selector('[role="menuitem"]', timeout=10000, state="visible")
        if menuitem:
            await menuitem.click()
            logger.info("ifd_nfhl_geocoder_result_selected")
    except PlaywrightTimeoutError:
        logger.warning("ifd_nfhl_geocoder_result_not_found")

    loop = asyncio.get_running_loop()
    started_at = loop.time()
    # Single wait against the union of popup selectors so a missing popup
    # costs one 15s timeout instead of one per candidate selector.
    popup_joined = ", ".join(NFHL_RESULT_POPUP_SELECTORS)
    try:
        popup_element = await page.wait_for_selector(popup_joined, timeout=15000, state="visible")
    except PlaywrightTimeoutError:
        popup_element = None
    popup_seen = popup_element is not None

    try:
        await page.wait_for_load_state("networkidle", timeout=30000)
    except PlaywrightTimeoutError as exc:
        logger.warning(f"ifd_nfhl_networkidle_timeout error={exc}")

    # Explicitly wait for the NFHL flood-overlay image(s) to finish decoding
    # before capturing. networkidle alone isn't enough on this slow site — the
    # overlay PNG can still be mid-paint (or have failed), which is the main
    # source of the corrupted captures. Best-effort: proceeds on timeout.
    images_ready = await _wait_for_map_images_rendered(page)

    # Generous tile-rasterization settle — Esri tiles continue to load after
    # networkidle reports zero in-flight requests because the canvas paints
    # asynchronously from the tile fetches.
    await asyncio.sleep(8)
    elapsed_ms = int((loop.time() - started_at) * 1000)

    if popup_seen:
        logger.info(f"ifd_nfhl_map_ready elapsed_ms={elapsed_ms} images_ready={images_ready}")
    else:
        logger.warning(
            f"ifd_nfhl_map_ready_no_popup elapsed_ms={elapsed_ms} images_ready={images_ready}"
        )


async def _capture_nfhl_page_pdf(page: Any, map_url: str) -> tuple[bytes, int]:
    """Capture the current NFHL viewer page as a Letter-landscape PDF.

    Reproduces the Chrome default "Headers and footers" print artifact the
    client's manual workflow produces:

    - header: print timestamp (left) + page title (right)
    - footer: viewer URL with ``&extent=`` (left) + page number (right)

    The timestamp and page number use Chromium's native print tokens
    (``class="date"`` and ``class="pageNumber"`` / ``class="totalPages"``), so
    the date is the actual print time formatted exactly as Chrome's manual
    Ctrl-P. The title is hard-coded (see ``NFHL_VIEWER_PAGE_TITLE``) so it
    matches even if headless ``document.title`` differs.

    The footer URL is the msc.fema.gov hand-off URL (``map_url``) printed
    literally — NOT Chromium's native ``class="url"`` token, which would emit
    ``page.url``: the Esri webappviewer rewrites ``window.location`` to a base
    URL during boot, dropping the ``&extent=`` coordinates.
    """
    await page.emulate_media(media="screen")

    # Flex row aligned to the body's 0.4in margins so header/footer text lines
    # up with the map edges, matching the manual print.
    row_style = (
        "font-size:9px; width:100%; padding:0 0.4in; box-sizing:border-box; "
        "display:flex; justify-content:space-between; align-items:center;"
    )
    header_html = (
        f'<div style="{row_style}">'
        '<span class="date" style="white-space:nowrap;"></span>'
        '<span style="white-space:nowrap; padding-left:12px;">'
        f"{escape(NFHL_VIEWER_PAGE_TITLE)}"
        "</span>"
        "</div>"
    )
    footer_html = (
        f'<div style="{row_style}">'
        '<span style="flex:1; min-width:0; white-space:nowrap; overflow:hidden; '
        f'text-overflow:ellipsis;">{escape(map_url)}</span>'
        '<span style="white-space:nowrap; padding-left:12px;">'
        '<span class="pageNumber"></span>/<span class="totalPages"></span>'
        "</span>"
        "</div>"
    )

    pdf_content = await page.pdf(
        format="Letter",
        landscape=True,
        print_background=True,
        display_header_footer=True,
        header_template=header_html,
        footer_template=footer_html,
        margin={"top": "0.4in", "right": "0.4in", "bottom": "0.4in", "left": "0.4in"},
    )
    pdf_size = len(pdf_content)
    logger.info(f"ifd_pdf_captured bytes={pdf_size}")
    return pdf_content, pdf_size


async def capture_fema_pdf_async(address: dict[str, str]) -> tuple[bytes, int]:
    """Capture the FEMA NFHL viewer map for an address as a PDF.

    Two-stage flow:
    1. **Hit msc.fema.gov** — search the address, extract the property-specific
       NFHL viewer URL from the "Go To NFHL Viewer" link, and (more importantly)
       acquire the `*.fema.gov` session cookies that the NFHL viewer's layer
       services check before serving FIRM Panels / NFHL data.
    2. **Open the NFHL viewer** with the extracted URL, dismiss any popups,
       search the address again in the geocoder, click the autocomplete result
       to commit the geocode, wait for tiles + overlays to render, capture PDF.

    Skipping stage 1 yields `credential is null` errors and the flood overlay
    never renders — that's what we were hitting before this rework.

    Args:
        address: Dict with at minimum `street`, `city`, `state`, `postal_code`.
            Optionally `unit` and `formatted` (a pre-built one-line address).

    Returns:
        Tuple of `(pdf_bytes, pdf_size)`. Caller is responsible for naming the
        file (the IFD agent uses `FLOODSEARCH.pdf`).
    """
    search_string = _build_search_string(address)
    if not search_string:
        raise ValueError("address has no usable street/city/state/postal_code values")

    logger.info(f"ifd_capture_start address='{search_string}'")

    async with async_playwright() as p:
        current_stage = "browser_launch"
        browser = None
        try:
            browser = await p.chromium.launch(headless=True, args=BROWSER_ARGS)
            page = await browser.new_page()
            await page.set_viewport_size({"width": 1440, "height": 1000})

            current_stage = "msc_fema"
            viewer_url = await _get_nfhl_viewer_url_from_msc_fema(page, search_string)
            if not viewer_url:
                # The msc.fema.gov stage couldn't extract the viewer URL, but
                # the session cookies it set may still be in the jar. Fall back
                # to the canonical viewer URL and hope the layers still load.
                logger.warning(f"ifd_msc_fema_fallback_to_canonical_url url={NFHL_VIEWER_URL}")
                viewer_url = NFHL_VIEWER_URL

            current_stage = "nfhl_nav"
            await _navigate_to_nfhl_viewer(page, viewer_url)
            current_stage = "nfhl_modals"
            await _dismiss_nfhl_modals(page)
            current_stage = "nfhl_search"
            await _search_nfhl_for_address(page, search_string)
            current_stage = "pdf_capture"
            # Footer URL = the property-positioned hand-off URL from msc.fema.gov
            # (already carries `&extent=`), NOT page.url (the viewer strips the
            # extent off window.location during boot).
            return await _capture_nfhl_page_pdf(page, viewer_url)

        except Exception as exc:
            # logger.exception preserves the traceback inline for diagnosis.
            logger.exception(f"ifd_capture_error stage={current_stage} error={exc}")
            raise
        finally:
            if browser is not None:
                try:
                    await browser.close()
                except Exception as close_exc:
                    # Don't let cleanup errors mask the primary exception.
                    logger.warning(f"ifd_browser_close_error error={close_exc}")
