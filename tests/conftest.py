"""Shared fixtures for IFD agent tests."""

from typing import Any

import pytest


@pytest.fixture
def single_property_loan() -> dict[str, Any]:
    """Loan details with a populated subject property address."""
    return {
        "loanNumber": "12345",
        "property": {
            "streetAddress": "1600 Pennsylvania Ave NW",
            "city": "Washington",
            "state": "DC",
            "postalCode": "20500",
        },
    }


@pytest.fixture
def missing_property_loan() -> dict[str, Any]:
    """Loan details with empty subject property block."""
    return {
        "loanNumber": "00000",
        "property": {},
    }


@pytest.fixture
def fake_bedrock_extraction_response() -> dict[str, Any]:
    """A minimally-shaped Bedrock Converse response carrying a flood zone tool_use block."""
    return {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "name": "extract_flood_zone",
                            "input": {
                                "flood_zone": "X",
                                "sfha": "no",
                                "panel_number": "12345C0123F",
                                "panel_effective_date": "05/16/2012",
                                "community_name": "Sample County",
                                "community_id": "120000",
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
