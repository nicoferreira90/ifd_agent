"""Utility functions for debugging and inspecting Encompass loan data."""

import json
import os
from typing import Any

from af.tools import logger


def dump_loan_details(
    details_json: dict[str, Any],
    loan_id: str,
    output_file: str | None = None,
    log_full_json: bool = False,
) -> str:
    """Print or save the full loan details JSON for debugging."""
    formatted = json.dumps(details_json, indent=2, default=str)

    logger.info(f"{'='*60}")
    logger.info(f"FULL LOAN DETAILS DEBUG for loan {loan_id}")
    logger.info(f"{'='*60}")

    top_level_keys = list(details_json.keys())
    logger.info(f"Top-level keys ({len(top_level_keys)} total):")
    for key in sorted(top_level_keys):
        value = details_json.get(key)
        value_type = type(value).__name__
        if isinstance(value, list):
            logger.info(f"   • {key}: [{value_type}] length={len(value)}")
        elif isinstance(value, dict):
            logger.info(f"   • {key}: [{value_type}] keys={list(value.keys())[:5]}...")
        elif isinstance(value, str) and len(value) > 50:
            logger.info(f"   • {key}: [{value_type}] '{value[:50]}...'")
        else:
            logger.info(f"   • {key}: [{value_type}] {value}")

    if "property" in details_json:
        prop = details_json.get("property", {})
        logger.info("\nProperty:")
        logger.info(f"   Keys: {list(prop.keys())}")
        logger.info(
            f"   Address: {prop.get('streetAddress', 'N/A')}, {prop.get('city', 'N/A')}, "
            f"{prop.get('state', 'N/A')} {prop.get('postalCode', 'N/A')}"
        )

    logger.info(f"\n{'='*60}")

    if output_file:
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                f.write(formatted)
            logger.info(f"Saved full JSON to: {output_file}")
        except Exception as e:
            logger.warning(f"Could not save JSON to file: {e}")

    if log_full_json:
        logger.debug(f"Full JSON:\n{formatted}")

    return formatted


def get_property_address_fields(details_json: dict[str, Any]) -> dict[str, Any]:
    """Extract property address fields from loan data for inspection."""
    result: dict[str, Any] = {
        "property_address": {},
        "subject_property": {},
    }

    prop = details_json.get("property", {}) or {}
    if prop:
        result["property_address"] = {
            "streetAddress": prop.get("streetAddress"),
            "city": prop.get("city"),
            "state": prop.get("state"),
            "postalCode": prop.get("postalCode"),
            "county": prop.get("county"),
        }

    subject = details_json.get("subjectProperty", {}) or {}
    if subject:
        result["subject_property"] = subject

    return result
