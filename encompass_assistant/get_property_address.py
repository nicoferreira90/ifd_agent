"""Subject property address lookup for IFD agent.

Reads the property fields off the Encompass loan JSON so the FEMA Map Service
Center search can be filled with the subject property address (the IFD pipeline
operates on the property, not on borrowers).
"""

from typing import Any

from af.tools import logger

from ifd_agent.encompass_assistant.exp_apis import get_loan_details

_STREET_KEYS = ("streetAddress", "addressLine1", "address1")
_CITY_KEYS = ("city",)
_STATE_KEYS = ("state", "stateCode", "propertyState")
_POSTAL_KEYS = ("postalCode", "zip", "zipCode")
_UNIT_KEYS = ("addressUnitIdentifier", "addressUnitDesignatorType", "unit", "addressUnit")


def _first_nonempty(d: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = d.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def get_property_address(
    access_token: str, api_server: str, loan_guid: str, loan_id: str = ""
) -> dict[str, str]:
    """Fetch the subject property address for the loan from Encompass.

    Args:
        access_token: Encompass OAuth bearer token.
        api_server: Encompass API server URL.
        loan_guid: Encompass loan GUID.
        loan_id: Loan ID (for logging only).

    Returns:
        Dict with `street`, `unit`, `city`, `state`, `postal_code` (all str,
        empty strings when missing) and a `formatted` single-line version
        suitable for filling into the FEMA search input.
    """
    details = get_loan_details(access_token, api_server, loan_guid)

    property_data = details.get("property") or {}
    if not isinstance(property_data, dict):
        property_data = {}

    street = _first_nonempty(property_data, *_STREET_KEYS)
    unit = _first_nonempty(property_data, *_UNIT_KEYS)
    city = _first_nonempty(property_data, *_CITY_KEYS)
    state = _first_nonempty(property_data, *_STATE_KEYS)
    postal_code = _first_nonempty(property_data, *_POSTAL_KEYS)

    parts: list[str] = []
    if street:
        parts.append(f"{street} {unit}".strip() if unit else street)
    locality_bits: list[str] = []
    if city:
        locality_bits.append(city)
    if state:
        locality_bits.append(state)
    if locality_bits:
        parts.append(", ".join(locality_bits))
    if postal_code:
        parts.append(postal_code)
    formatted = ", ".join(parts) if parts else ""

    logger.info(
        f"ifd_get_property_address loan_id={loan_id} street='{street}' unit='{unit}' "
        f"city='{city}' state='{state}' postal_code='{postal_code}' formatted='{formatted}'"
    )

    return {
        "street": street,
        "unit": unit,
        "city": city,
        "state": state,
        "postal_code": postal_code,
        "formatted": formatted,
    }
