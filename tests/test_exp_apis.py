"""Tests for get_custom_field_value in the IFD encompass_assistant.exp_apis module.

Regression coverage for a fieldReader response-shape bug: Encompass's
fieldReader endpoint has been observed to return either the
`[{"id": ..., "value": ...}, ...]` list shape or a `{"<field_id>": "<value>"}`
dict shape (see bk_search_agent's `_extract_field_value_from_field_reader_response`
for the same ambiguity handled there). Only the list shape was originally
handled here, which silently returned "" for a dict-shaped response.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from ifd_agent.encompass_assistant.exp_apis import get_custom_field_value


def _mock_response(status_code: int, json_body: Any) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_body
    return response


class TestGetCustomFieldValue:
    def test_extracts_value_from_list_shape(self) -> None:
        with patch("ifd_agent.encompass_assistant.exp_apis.requests") as mock_requests:
            mock_requests.request.return_value = _mock_response(
                200, [{"id": "CX.INITPROCNOTES", "value": "some note"}]
            )
            result = get_custom_field_value(
                "token", "https://encompass.example", "guid", "12345", "CX.INITPROCNOTES"
            )
        assert result == "some note"

    def test_extracts_value_from_dict_shape(self) -> None:
        with patch("ifd_agent.encompass_assistant.exp_apis.requests") as mock_requests:
            mock_requests.request.return_value = _mock_response(
                200, {"CX.INITPROCNOTES": "some note"}
            )
            result = get_custom_field_value(
                "token", "https://encompass.example", "guid", "12345", "CX.INITPROCNOTES"
            )
        assert result == "some note"

    def test_returns_empty_string_when_field_absent_from_list(self) -> None:
        with patch("ifd_agent.encompass_assistant.exp_apis.requests") as mock_requests:
            mock_requests.request.return_value = _mock_response(200, [{"id": "1387", "value": "X"}])
            result = get_custom_field_value(
                "token", "https://encompass.example", "guid", "12345", "CX.INITPROCNOTES"
            )
        assert result == ""

    def test_returns_empty_string_when_field_absent_from_dict(self) -> None:
        with patch("ifd_agent.encompass_assistant.exp_apis.requests") as mock_requests:
            mock_requests.request.return_value = _mock_response(200, {"1387": "X"})
            result = get_custom_field_value(
                "token", "https://encompass.example", "guid", "12345", "CX.INITPROCNOTES"
            )
        assert result == ""

    def test_returns_empty_string_on_non_200(self) -> None:
        with patch("ifd_agent.encompass_assistant.exp_apis.requests") as mock_requests:
            mock_requests.request.return_value = _mock_response(500, {})
            result = get_custom_field_value(
                "token", "https://encompass.example", "guid", "12345", "CX.INITPROCNOTES"
            )
        assert result == ""

    def test_returns_empty_string_on_unexpected_payload_shape(self) -> None:
        with patch("ifd_agent.encompass_assistant.exp_apis.requests") as mock_requests:
            mock_requests.request.return_value = _mock_response(200, "not a list or dict")
            result = get_custom_field_value(
                "token", "https://encompass.example", "guid", "12345", "CX.INITPROCNOTES"
            )
        assert result == ""

    def test_returns_empty_string_on_request_exception(self) -> None:
        with patch("ifd_agent.encompass_assistant.exp_apis.requests") as mock_requests:
            mock_requests.request.side_effect = RuntimeError("connection failed")
            result = get_custom_field_value(
                "token", "https://encompass.example", "guid", "12345", "CX.INITPROCNOTES"
            )
        assert result == ""
