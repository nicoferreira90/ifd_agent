"""Tests for request parsing helpers used by the IFD agent entrypoint and tool."""

from ifd_agent.encompass_mcp_tool import EncompassMCPTool
from ifd_agent.ifd_agent import extract_loan_id_from_prompt


def test_extract_loan_id_from_prompt_handles_loan_number_phrase() -> None:
    assert extract_loan_id_from_prompt("Process loan 87025103184 for IFD portal") == "87025103184"


def test_extract_loan_id_from_prompt_handles_standalone_number() -> None:
    assert extract_loan_id_from_prompt("87025103184") == "87025103184"


def test_extract_loan_id_from_prompt_preserves_dev_suffix() -> None:
    assert extract_loan_id_from_prompt("Process loan 87025103184-dev for IFD") == "87025103184-dev"


def test_extract_loan_id_from_prompt_returns_unknown_when_missing() -> None:
    assert extract_loan_id_from_prompt("Run a flood determination, please") == "unknown"


def test_mcp_tool_parses_loan_id_and_defaults_portal_to_ifd() -> None:
    tool = EncompassMCPTool()

    parsed = tool._parse_user_input("Run flood determination for loan 12345")

    assert parsed is not None
    assert parsed["loan_id"] == "12345"
    assert parsed["portal"] == "IFD"


def test_mcp_tool_returns_none_when_no_loan_id_present() -> None:
    tool = EncompassMCPTool()

    parsed = tool._parse_user_input("Tell me about flood zones")

    assert parsed is None
