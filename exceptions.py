"""Custom exceptions for the IFD agent."""


class LoanLockConflictError(Exception):
    """Raised when a loan is locked by another user (HTTP 409).

    This is a non-retryable error — retrying will not help until the
    other user releases the lock.
    """

    def __init__(self, loan_guid: str, locked_by: str, raw_response: str = ""):
        self.loan_guid = loan_guid
        self.locked_by = locked_by
        self.raw_response = raw_response
        super().__init__(f"Loan '{loan_guid}' is currently locked by another user '{locked_by}'")
