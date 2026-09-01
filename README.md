# IFD Agent

AI-powered agent for automating Initial Flood Determination (IFD) lookups on the FEMA Map Service Center.

## Overview

The IFD Agent automates the manual Initial Flood Determination pipeline:

1. Read the subject property address from the Encompass loan record.
2. Drive `msc.fema.gov` with Playwright, type the address into the FEMA search bar, and capture the rendered map page as `FLOODSEARCH.pdf`.
3. Render the captured PDF to PNG with PyMuPDF and ask a Bedrock vision call (forced `toolSpec`) to extract the FEMA flood zone, SFHA status, FIRM panel number/effective date, community name, and community ID.
4. Write the extracted flood zone to Encompass standard field **1387** (Flood Zone).
5. Upload `FLOODSEARCH.pdf` to Encompass eFolder bucket **132 - Flood Search**.

## Features

- **FEMA Map Service Center automation** — Playwright fills the address search bar and waits for the map + SFHA panel to render before capturing the PDF.
- **AI-driven flood zone extraction** — a Bedrock vision model with a strict JSON-schema `toolSpec` parses the captured map; the `zone_found` field gates downstream writes.
- **Encompass integration** — Writes field 1387 first (fail safely if the field write fails), then uploads `FLOODSEARCH.pdf` to bucket 132.
- **Eligibility & expiration checks** — Skips DFT/CANCELED/DENIED loans and any loan whose bucket 132 already has a document under 30 days old.

## Workflow

```
loan_id
  -> connect to Encompass
  -> eligibility check (DFT/CANCELED/DENIED, state, loan type)
  -> 30-day expiration check on bucket "132 - Flood Search"
  -> get_property_address (loan.property.{streetAddress,city,state,postalCode})
  -> capture_fema_pdf_async(address)   (Playwright -> FLOODSEARCH.pdf)
  -> extract_flood_zone(pdf_bytes)     (PyMuPDF -> Bedrock vision)
  -> if zone_found != "yes": short-circuit with status=needs_review
  -> update_custom_fields(field 1387 = <flood_zone>)
  -> upload_file_into_efolder(portal="IFD")
```

## Directory Structure

```
ifd_agent/
├── ifd_agent.py                  # Main agent entry point (AgentCore + Strands)
├── encompass_mcp_tool.py         # Strands tools (process_encompass_request, ...)
├── encompass_functions.py        # Core single-property processing flow
├── playwright_assistant/
│   └── capturePDF_async.py       # FEMA portal Playwright automation
├── vision_assistant/
│   └── flood_zone_extractor.py   # PyMuPDF + Bedrock vision tool-use
├── encompass_assistant/
│   ├── exp_apis.py               # Encompass API helpers
│   ├── get_property_address.py   # Subject property address lookup
│   ├── get_connection.py         # Encompass connection
│   └── upload_file.py            # eFolder upload
├── utils/                        # Shared helpers (misc, s3, loan-details debug)
├── models.py                     # Pydantic response models
├── process_tracker.py            # Step-by-step process tracker
├── efolder_mapping.json          # Portal -> bucket mapping (IFD -> 132)
├── requirements.txt              # Agent-specific deps (pymupdf, pypdf, ...)
├── Dockerfile                    # Multi-stage Playwright + af-tools image
└── tests/                        # Smoke tests
```

## Usage

```json
{ "prompt": "Process loan 87025103184 for IFD portal" }
```

Or, equivalent shorthand:

```json
{ "prompt": "87025103184" }
```

### Example Response

```json
{
  "status": "success",
  "loan_id": "87025103184",
  "request_id": "abc-123",
  "processes": [
    {
      "code": "ifd",
      "execution_state_code": "completed",
      "steps": [
        {"code": "get-property-address", "execution_state_code": "completed"},
        {"code": "search-website-fema", "execution_state_code": "completed"},
        {"code": "capture-website-pdf-fema", "execution_state_code": "completed"},
        {"code": "extract-flood-zone-vision", "execution_state_code": "completed"},
        {"code": "write-flood-zone-field", "execution_state_code": "completed"},
        {"code": "upload-to-encompass-efolder-132-flood", "execution_state_code": "completed"}
      ]
    }
  ],
  "response": "..."
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_REGION` | AWS region | `us-east-1` |
| `BEDROCK_MODEL_ID` | Bedrock model for the main Strands agent | (set per env) |
| `FLOOD_ZONE_EXTRACTOR_MODEL_ID` | Bedrock model used by the vision extractor | `us.anthropic.claude-sonnet-4-6` |
| `PLATFORM_SECRETS_ARN` | AWS Secrets Manager ARN with Encompass credentials | (required) |
| `RUNTIME_NAME` | AgentCore runtime ID used for CloudWatch log group | `ifd-agent` |

## Behavior choices baked in for v1

- **Single PDF capture path** — direct async Playwright only, no MCP-playwright fallback.
- **Vision is the only zone extractor** — no DOM-scrape fast-path on FEMA's portal.
- **`zone_found != "yes"` -> short-circuit** — return `status=needs_review` with the extracted dict and skip both the field write and the eFolder upload.
- **Field write before eFolder upload** — if PATCH on field 1387 fails we don't dirty bucket 132 with a half-completed determination.

## Open questions (track during iteration)

- Confirm field `1387` matches the dev Encompass instance; swap if your config uses a `CX.*` custom field.
- FEMA search input selectors may need adjustment after the portal re-skins; the candidate list in `playwright_assistant/capturePDF_async.py` is intentionally short.
- Bump `DEFAULT_DPI` from 150 to 200 in `vision_assistant/flood_zone_extractor.py` if the model misreads the zone label.
- Wire `panel_number`, `panel_effective_date`, `community_name`, and `community_id` to Encompass fields once the IDs are confirmed (likely `1388`, `1395`, etc.).

<!-- Trigger rebuild -->
