"""Text utility functions for IFD agent."""


def mask_ssn(ssn: str) -> str:
    """Mask SSN showing only last 4 digits.

    Args:
        ssn: Social Security Number (9 digits).

    Returns:
        Masked SSN in format "***-**-XXXX" or "N/A" if invalid.
    """
    if ssn and len(ssn) >= 4:
        return f"***-**-{ssn[-4:]}"
    return "N/A"
