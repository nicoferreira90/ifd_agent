import datetime
import json
import re
from typing import Any
from urllib.parse import unquote as url_decode

import requests
from af.tools import logger
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_fixed

from ifd_agent.exceptions import LoanLockConflictError

CONTENT_TYPE_PDF = "application/pdf"
CONTENT_TYPE_JSON = "application/json"


@retry(stop=stop_after_attempt(3), reraise=True, wait=wait_fixed(5))
def _get_access_token(
    api_server: str,
    instance_id: str,
    api_user_client_id: str,
    api_user_client_secret: str,
    encompass_username: str,
    encompass_password: str,
) -> str:
    url = f"{api_server}/oauth2/v1/token"

    payload = {
        "grant_type": "password",
        "username": f"{encompass_username}@encompass:{instance_id}",
        "password": encompass_password,
        "client_id": api_user_client_id,
        "client_secret": api_user_client_secret,
    }

    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    access_token = None
    try:
        response = requests.request("POST", url, headers=headers, data=payload)

        if response.status_code == 200:
            json_dict = json.loads(response.text)
            access_token = json_dict.get("access_token", None)
        else:
            logger.info(response.text)
            raise ValueError(f"Unable to get access_token, status_code = {response.status_code}")

    except Exception as exc:
        logger.exception(f"Unable to get access_token {exc}")
        raise ValueError("Unable to get access_token") from exc

    if not access_token:
        raise ValueError("Access token is None")
    if not isinstance(access_token, str):
        raise ValueError(f"Access token is not a string, got {type(access_token)}")
    return access_token


@retry(stop=stop_after_attempt(3), reraise=True, wait=wait_fixed(5))
def _get_attachment_upload_url(
    api_server: str,
    loan_guid: str,
    file_size: int,
    file_name: str,
    access_token: str,
    entity_id: str,
    loan_id: str = "",
) -> tuple[str | None, str | None, bool, dict[str, Any] | None]:

    url = f"{api_server}/encompass/v3/loans/{loan_guid}/attachmentUploadUrl"
    payload = {
        "file": {"contentType": CONTENT_TYPE_PDF, "name": file_name, "size": file_size},
        "assignTo": {"entityId": entity_id, "entityType": "Document"},
        "title": file_name,
    }

    if not entity_id:
        logger.info(f"{loan_id} {file_name} {entity_id} entity id is null ")
        del payload["assignTo"]

    payload_json = json.dumps(payload)

    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": CONTENT_TYPE_JSON}

    upload_url = None
    authorization_header = None
    multichunk_required = False
    multichunk_json = None

    try:
        response = requests.request("POST", url, headers=headers, data=payload_json, timeout=30)

        logger.info(
            f"{loan_id} {file_name} {entity_id} attachmentUploadUrl status_code {response.status_code}"
        )

        if response.status_code != 200:
            logger.info(
                f"{loan_id} {file_name} {entity_id} Unable to get upload url loan_guid {loan_guid} status_code: {response.status_code} Re-queueing message"
            )
            raise ValueError(response.text)

        json_dict = json.loads(response.text)

        if json_dict.get("multiChunkRequired", None):
            multichunk_required = True
            multichunk_json = json_dict
        else:
            upload_url = json_dict.get("uploadUrl", None)
            authorization_header = json_dict.get("authorizationHeader", None)

    except Exception as e:
        logger.info(
            f"{loan_id} {file_name} {entity_id} exception to get upload url Re-queueing message"
        )
        raise e

    return upload_url, authorization_header, multichunk_required, multichunk_json


@retry(stop=stop_after_attempt(3), reraise=True, wait=wait_fixed(5))
def _upload_attachment_multichunk(
    multichunk_json: dict[str, Any], file_obj: Any, file_name: str = "", loan_id: str = ""
) -> None:

    chunk_start = 0
    chunk_end = 0
    call_commit_url = True

    authorization_header = multichunk_json["authorizationHeader"]

    logger.info(
        f"{loan_id} {file_name} Total number of chunk: {multichunk_json['multiChunk']['chunkList']}"
    )

    for chunks in multichunk_json["multiChunk"]["chunkList"]:
        chunk_end = chunk_start + chunks["size"]
        chunked_byte = file_obj[chunk_start:chunk_end]
        chunk_start = chunk_end

        url = chunks["uploadUrl"]

        headers = {"Authorization": authorization_header, "Content-Type": CONTENT_TYPE_PDF}

        response = None
        try:
            response = requests.request(
                "PUT", url_decode(url), headers=headers, data=chunked_byte, timeout=60
            )

            logger.info(f"{loan_id} {file_name} response statusCode {response.status_code}")

            if response.status_code != 200:
                call_commit_url = False
                logger.info(f"{loan_id} {file_name} Upload response = {response.text}")

        except Exception as e:
            logger.exception(f"{loan_id} {file_name} exception while uploading file {e}")
            raise ValueError(f"{loan_id} {file_name} exception while uploading file") from e

    if call_commit_url:

        url = multichunk_json["multiChunk"]["commitUrl"]

        try:
            response = requests.request("POST", url, headers=headers, timeout=30)

            logger.info(f"{loan_id} {file_name} commitUrl {response.text} {response.status_code}")

        except Exception as e:
            logger.exception(f"{loan_id} {file_name} exception while committing file {e}")
            raise ValueError(f"{loan_id} {file_name} exception while committing file") from e


@retry(stop=stop_after_attempt(3), reraise=True, wait=wait_fixed(5))
def upload_attachment(
    access_token: str,
    api_server: str,
    loan_guid: str,
    file_size: int,
    file_name: str,
    file_obj: Any,
    entity_id: str,
    loan_id: str = "",
) -> int | None:

    url, authorization_header, multichunk_required, multichunk_json = None, None, None, None
    try:
        url, authorization_header, multichunk_required, multichunk_json = (
            _get_attachment_upload_url(
                api_server, loan_guid, file_size, file_name, access_token, entity_id, loan_id
            )
        )

        if multichunk_required:

            logger.info(
                f"{loan_id} {file_name} {entity_id} multichunk_required {multichunk_required}"
            )

            try:
                if multichunk_json is None:
                    raise ValueError("Multichunk JSON is None")
                _upload_attachment_multichunk(multichunk_json, file_obj, file_name, loan_id)
                return 0
            except Exception as e:

                logger.exception(
                    f"{loan_id} {file_name} {entity_id} exception while uploading multichunk file {e}"
                )
                raise ValueError(
                    f"{loan_id} {file_name} {entity_id} exception while uploading multichunk file"
                ) from e
        else:

            if url is None or authorization_header is None:
                raise ValueError(
                    f"{loan_id} {file_name} {entity_id} Unable to get upload url or header"
                )

    except Exception as e:
        logger.exception(f"{loan_id} {file_name} {entity_id} exception while getting upload url")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception message: {str(e)}")
        logger.error(f"API Server: {api_server}")
        logger.error(f"Loan GUID: {loan_guid}")
        logger.error(f"Entity ID: {entity_id}")
        logger.error(f"File name: {file_name}, Size: {file_size}")

        error_msg = (
            f"{loan_id} {file_name} {entity_id} exception while getting upload url: {str(e)}"
        )
        raise ValueError(error_msg) from e

    payload = file_obj
    headers = {"Authorization": authorization_header, "Content-Type": CONTENT_TYPE_PDF}

    response = None
    try:
        response = requests.request(
            "PUT", url_decode(url), headers=headers, data=payload, timeout=60
        )
    except Exception:
        logger.info(f"{loan_id} {file_name} {entity_id} exception while uploading file ")
        raise

    if response is not None and response.status_code != 200:
        raise ValueError(
            f"{loan_id} {file_name} {entity_id} Unable to upload file to eFolder status_code = {response.status_code} Re-queueing message"
        )

    logger.info(f"{loan_id} {file_name} Upload response = {response}")
    return None


@retry(
    stop=stop_after_attempt(3),
    reraise=True,
    wait=wait_fixed(5),
    retry=retry_if_not_exception_type(LoanLockConflictError),
)
def update_custom_fields(
    access_token: str,
    api_server: str,
    loan_guid: str,
    loan_id: str,
    fields_map_list: list[dict[str, Any]],
) -> None:
    """Write Encompass loan fields via the v3 fieldWriter endpoint.

    Accepts a list of `{"id": "<field>", "value": "<value>"}` dicts and POSTs
    them to `/encompass/v3/loans/{guid}/fieldWriter`. Field IDs can be either
    standard numeric Encompass fields (e.g. `1387` for Flood Zone) or custom
    `CX.*` fields.
    """

    logger.info(f"Updating custom fields {loan_id} payload {json.dumps(fields_map_list)}")

    response = None
    try:

        url = f"{api_server}/encompass/v3/loans/{loan_guid}/fieldWriter"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": CONTENT_TYPE_JSON,
        }

        payload = json.dumps(fields_map_list)

        response = requests.request("POST", url, headers=headers, data=payload, timeout=30)
        logger.info(
            f"custom loan status {loan_id} Status Code: {response.status_code} {response.text} payload {payload}"
        )
        if (response.status_code) // 100 != 2:
            if response.status_code == 409:
                _raise_if_lock_conflict(response, loan_guid)
            logger.info(
                f"custom loan status raised exception {loan_id} {response.text} Status Code {response.status_code} payload {payload}"
            )
            raise ValueError(
                f"custom loan status raised exception {loan_id}: "
                f"{response.status_code} {response.text}"
            )

    except Exception as e:
        if response is not None:
            logger.info(
                f"custom loan status exception {loan_id} {response.text} Status Code {response.status_code} {str(e)}"
            )
        else:
            logger.info(f"custom loan status exception {loan_id} (no response): {str(e)}")
        raise


def get_custom_field_value(
    access_token: str,
    api_server: str,
    loan_guid: str,
    loan_id: str,
    field_id: str,
) -> str:
    """Read a single Encompass field's current value via the v3 fieldReader endpoint.

    fieldReader has been observed to return either the `[{"id": ..., "value":
    ...}, ...]` list shape or a `{"<field_id>": "<value>"}` dict shape (see
    `bk_search_agent`'s `_extract_field_value_from_field_reader_response`,
    which handles the same ambiguity) — both are normalized here.

    Best-effort: returns "" on any non-200 response, unexpected payload shape,
    or request failure, rather than raising. Callers that use this as a guard
    before a destructive write should treat "" the same as "field is unset"
    so a read failure never causes a value to be dropped.
    """
    url = f"{api_server}/encompass/v3/loans/{loan_guid}/fieldReader"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": CONTENT_TYPE_JSON,
    }
    try:
        response = requests.request(
            "POST", url, headers=headers, data=json.dumps([field_id]), timeout=30
        )
        if response.status_code != 200:
            logger.warning(
                f"get_custom_field_value loan_id={loan_id} field_id={field_id} "
                f"status_code={response.status_code}"
            )
            return ""
        fields_data = response.json()
        if isinstance(fields_data, dict):
            return str(fields_data.get(field_id) or "").strip()
        if not isinstance(fields_data, list):
            return ""
        for field in fields_data:
            if isinstance(field, dict) and field.get("id") == field_id:
                return str(field.get("value") or "").strip()
        return ""
    except Exception as exc:
        logger.warning(f"get_custom_field_value loan_id={loan_id} field_id={field_id} error={exc}")
        return ""


def get_loan_status(api_server: str, access_token: str, loan_id: str, loan_guid: str) -> None:

    url = f"{api_server}/encompass/v3/loans/{loan_guid}/fieldReader"
    payload = json.dumps(["CX.DASHUPLOAD.01", "CX.DASHUPLOAD.02", "CX.DASHUPLOAD.03"])
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": CONTENT_TYPE_JSON,
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    logger.info(f"{loan_id} {response.text}")


@retry(stop=stop_after_attempt(3), reraise=True, wait=wait_fixed(5))
def check_document_expiration(
    access_token: str,
    api_server: str,
    loan_guid: str,
    loan_id: str,
    bucket_name: str,
    days_threshold: int = 30,
) -> dict[str, Any]:
    """Check if a document in a specific eFolder bucket is expired."""
    try:
        documents = get_all_retrieve_documents(access_token, api_server, loan_guid)

        bucket_documents = []
        bucket_name_lower = bucket_name.strip().lower()

        for doc in documents:
            doc_title = doc.get("title", "").strip().lower()
            if bucket_name_lower in doc_title:
                attachments = doc.get("attachments", [])
                if attachments:
                    for attachment in attachments:
                        attachment_date = attachment.get("dateCreated") or attachment.get(
                            "dateModified"
                        )
                        if attachment_date:
                            bucket_documents.append(
                                {
                                    "title": doc.get("title"),
                                    "attachment_date": attachment_date,
                                    "attachment_id": attachment.get("id"),
                                    "document_id": doc.get("id"),
                                }
                            )

        if not bucket_documents:
            logger.info(
                f"No documents found in bucket '{bucket_name}' for loan {loan_id} - proceeding with processing"
            )
            return {
                "has_document": False,
                "document_date": None,
                "days_old": None,
                "is_expired": True,
                "skip_processing": False,
            }

        most_recent = max(bucket_documents, key=lambda x: x.get("attachment_date", ""))
        document_date_str = most_recent.get("attachment_date")

        document_date = None
        try:
            if document_date_str and "T" in document_date_str:
                date_part = document_date_str.split("T")[0]
                time_part = (
                    document_date_str.split("T")[1].split("+")[0].split("Z")[0].split("-")[0]
                )
                try:
                    document_date = datetime.datetime.fromisoformat(
                        document_date_str.replace("Z", "+00:00")
                    )
                except Exception:
                    document_date = datetime.datetime.strptime(
                        f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S"
                    )
            elif document_date_str and len(document_date_str) >= 10:
                document_date = datetime.datetime.strptime(document_date_str[:10], "%Y-%m-%d")
            else:
                raise ValueError(f"Unknown date format: {document_date_str}")
        except Exception as e:
            logger.warning(
                f"Could not parse document date '{document_date_str}' for loan {loan_id}, bucket '{bucket_name}': {e}"
            )
            return {
                "has_document": True,
                "document_date": document_date_str,
                "days_old": None,
                "is_expired": True,
                "skip_processing": False,
            }

        if document_date.tzinfo:
            document_date = document_date.replace(tzinfo=None)

        current_date = datetime.datetime.now()
        days_old = (current_date - document_date).days

        is_expired = days_old >= days_threshold

        logger.info(
            f"Document in bucket '{bucket_name}' for loan {loan_id}: {days_old} days old, expired={is_expired}, threshold={days_threshold} days"
        )

        return {
            "has_document": True,
            "document_date": document_date_str,
            "days_old": days_old,
            "is_expired": is_expired,
            "skip_processing": not is_expired,
        }

    except Exception as e:
        logger.exception(
            f"Error checking document expiration for loan {loan_id}, bucket '{bucket_name}': {e}"
        )
        return {
            "has_document": False,
            "document_date": None,
            "days_old": None,
            "is_expired": True,
            "skip_processing": False,
        }


def _check_whats_up_eligibility(
    whats_up: Any, whats_up_field_id: str, loan_status: str | None, loan_id: str
) -> dict[str, Any]:
    """Check if a loan is eligible based on the "What's Up?" field value."""
    is_eligible = True
    exclusion_reason = None
    if whats_up:
        whats_up_upper = whats_up.upper() if isinstance(whats_up, str) else str(whats_up).upper()
        exclusion_keywords = ["DFT", "CANCELED", "DENIED", "CANCELLED", "DEAL FELL THROUGH"]
        for keyword in exclusion_keywords:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, whats_up_upper):
                is_eligible = False
                exclusion_reason = f"'What's Up?' field contains: {keyword}"
                logger.warning(
                    f"Loan {loan_id} excluded - 'What's Up?' contains: {keyword} (from field: {whats_up_field_id})"
                )
                break
    whats_up_value = whats_up if whats_up_field_id == "CUST01FV" else (whats_up or "")
    return {
        "status": loan_status or "Unknown",
        "whats_up": whats_up_value,
        "is_eligible": is_eligible,
        "exclusion_reason": exclusion_reason,
    }


@retry(stop=stop_after_attempt(3), reraise=True, wait=wait_fixed(5))
def get_loan_status_and_whats_up(
    access_token: str, api_server: str, loan_guid: str, loan_id: str
) -> dict[str, Any]:
    """Get loan status and "What's Up?" field to verify loan is eligible for processing."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": CONTENT_TYPE_JSON,
    }

    loan_status = None
    whats_up = None
    status_field_id = None
    whats_up_field_id = None

    try:
        logger.info(
            f"Loan {loan_id} - Attempting to get full loan data to read CUST01FV from customFields..."
        )
        try:
            loan_url = f"{api_server}/encompass/v3/loans/{loan_guid}"
            loan_response = requests.get(loan_url, headers=headers, timeout=30)

            if loan_response.status_code == 200:
                loan_data = loan_response.json()

                if "customFields" in loan_data and isinstance(loan_data["customFields"], list):
                    for field in loan_data["customFields"]:
                        if isinstance(field, dict) and field.get("fieldName") == "CUST01FV":
                            field_value = field.get("value") or field.get("stringValue")
                            if field_value:
                                whats_up = str(field_value).strip()
                                whats_up_field_id = "CUST01FV"
                                logger.info(
                                    f"Loan {loan_id} - Found 'What's Up?' (CUST01FV) in customFields: '{whats_up}'"
                                )
                                break

                if not loan_status:
                    status_candidates = [
                        loan_data.get("loanStatus"),
                        loan_data.get("status"),
                        loan_data.get("loan_status"),
                    ]
                    for candidate in status_candidates:
                        if candidate:
                            loan_status = str(candidate).upper().strip()
                            status_field_id = "loan_data"
                            logger.info(
                                f"Loan {loan_id} - Found status from loan data: {loan_status} (from field: {status_field_id})"
                            )
                            break
        except Exception as loan_data_error:
            logger.warning(f"Loan {loan_id} - Could not get full loan data: {loan_data_error}")

        url = f"{api_server}/encompass/v3/loans/{loan_guid}/fieldReader"

        fields_to_read = [
            "Loan.LoanStatus",
            "Loan.LoanFolder",
            "14",
            "1172",
        ]

        if not whats_up:
            fields_to_read.extend(
                [
                    "Loan.WhatsUp",
                    "CX.WHATSUP",
                ]
            )

        payload = json.dumps(fields_to_read)
        response = requests.request("POST", url, headers=headers, data=payload, timeout=30)

        if response.status_code == 200:
            try:
                fields_data = response.json()
            except json.JSONDecodeError as json_error:
                error_msg = f"Invalid JSON response from Encompass API for loan {loan_id}"
                logger.error(f"{error_msg}: {json_error}")
                return {
                    "status": "Unknown",
                    "whats_up": "",
                    "is_eligible": True,
                    "exclusion_reason": None,
                    "error": {
                        "occurred": True,
                        "type": "validation_error",
                        "message": error_msg,
                        "details": {"json_error": str(json_error)},
                    },
                }

            if not isinstance(fields_data, list):
                error_msg = f"Unexpected response structure for loan {loan_id}"
                logger.error(error_msg)
                return {
                    "status": "Unknown",
                    "whats_up": "",
                    "is_eligible": True,
                    "exclusion_reason": None,
                    "error": {
                        "occurred": True,
                        "type": "validation_error",
                        "message": error_msg,
                        "details": {
                            "expected_type": "list",
                            "actual_type": type(fields_data).__name__,
                        },
                    },
                }

            logger.info(
                f"Loan {loan_id} - Field reader response: {json.dumps(fields_data, indent=2)}"
            )

            if not loan_status:
                for field in fields_data:
                    if not isinstance(field, dict):
                        continue
                    field_id = field.get("id", "")
                    field_value = field.get("value", "")
                    if "status" in field_id.lower() and field_value:
                        loan_status = str(field_value).upper().strip()
                        logger.info(
                            f"Loan {loan_id} - Found status: {loan_status} from field {field_id}"
                        )
                        break

            if not whats_up:
                for field in fields_data:
                    if not isinstance(field, dict):
                        continue
                    field_id = field.get("id", "")
                    field_value = field.get("value", "")
                    if "whatsup" in field_id.lower() or "what's up" in field_id.lower():
                        whats_up = str(field_value).upper().strip() if field_value else ""
                        whats_up_field_id = field_id
                        logger.info(
                            f"Loan {loan_id} - Found 'What's Up?': {whats_up} from field {field_id}"
                        )
                        break

            is_eligible = True
            exclusion_reason = None

            if whats_up:
                whats_up_upper = (
                    whats_up.upper() if isinstance(whats_up, str) else str(whats_up).upper()
                )
                exclusion_keywords = ["DFT", "CANCELED", "DENIED", "CANCELLED", "DEAL FELL THROUGH"]

                for keyword in exclusion_keywords:
                    pattern = r"\b" + re.escape(keyword) + r"\b"
                    if re.search(pattern, whats_up_upper):
                        is_eligible = False
                        exclusion_reason = f"'What's Up?' field contains: {keyword}"
                        logger.warning(
                            f"Loan {loan_id} excluded - 'What's Up?' contains: {keyword}"
                        )
                        break

            property_state = None
            loan_type_value = None
            for field in fields_data:
                if not isinstance(field, dict):
                    continue
                fid = field.get("id", "")
                fval = field.get("value", "")
                if fid == "14" and fval:
                    property_state = str(fval).strip()
                elif fid == "1172" and fval:
                    loan_type_value = str(fval).strip()

            if is_eligible and property_state and property_state.upper() in {"NH"}:
                is_eligible = False
                exclusion_reason = f"Property state '{property_state}' is excluded from processing"
                logger.warning(f"Loan {loan_id} excluded - property state: {property_state}")
            if is_eligible and loan_type_value and loan_type_value.upper() in {"USDA-RD"}:
                is_eligible = False
                exclusion_reason = f"Loan type '{loan_type_value}' is excluded from processing"
                logger.warning(f"Loan {loan_id} excluded - loan type: {loan_type_value}")

            whats_up_value = whats_up if whats_up_field_id == "CUST01FV" else (whats_up or "")

            return {
                "status": loan_status or "Unknown",
                "whats_up": whats_up_value,
                "is_eligible": is_eligible,
                "exclusion_reason": exclusion_reason,
                "error": None,
            }

        else:
            eligibility_result = _check_whats_up_eligibility(
                whats_up, whats_up_field_id or "", loan_status, loan_id
            )
            return {
                **eligibility_result,
                "error": {
                    "occurred": True,
                    "type": "api_error",
                    "message": f"HTTP {response.status_code} for loan {loan_id}",
                    "details": {
                        "status_code": response.status_code,
                        "response": response.text[:200],
                    },
                },
            }

    except requests.exceptions.Timeout as timeout_error:
        error_msg = f"Request timeout while reading loan status for {loan_id}"
        logger.error(f"{error_msg}: {timeout_error}")
        eligibility_result = _check_whats_up_eligibility(
            whats_up, whats_up_field_id or "", loan_status, loan_id
        )
        return {
            **eligibility_result,
            "error": {
                "occurred": True,
                "type": "timeout_error",
                "message": error_msg,
                "details": {"timeout_seconds": 30, "error": str(timeout_error)},
            },
        }

    except Exception as e:
        error_msg = f"Unexpected error reading loan status for {loan_id}"
        logger.exception(f"{error_msg}: {e}")
        eligibility_result = _check_whats_up_eligibility(
            whats_up, whats_up_field_id or "", loan_status, loan_id
        )
        return {
            **eligibility_result,
            "error": {
                "occurred": True,
                "type": "unknown_error",
                "message": error_msg,
                "details": {"error": str(e), "error_type": type(e).__name__},
            },
        }


@retry(stop=stop_after_attempt(3), reraise=True)
def _get_loan_guid(access_token: str, api_server: str, loan_id: str) -> str:

    url = f"{api_server}/encompass/v3/loanPipeline"

    payload = json.dumps(
        {
            "filter": {"canonicalName": "Loan.LoanNumber", "value": loan_id, "matchType": "exact"},
            "orgType": "Internal",
            "loanOwnership": "AllLoans",
        }
    )

    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    logger.info(
        f"_get_loan_guid from loan id url: {url} payload: {payload} access_token: [REDACTED]"
    )

    load_guid = None

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        json_dict = json.loads(response.text)
        logger.info(f"json_dict: {json_dict}")

        if len(json_dict) and json_dict[0].get("loanId", None):
            load_guid = json_dict[0]["loanId"]
        else:
            logger.info(f"response _get_loan_guid {response.text}")
    except Exception as e:
        logger.exception(f"Unable to fetch loan guid {e}")
        raise ValueError("Unable to fetch loan guid from econnect") from e

    if not load_guid:
        raise ValueError("Loan GUID not found in response")
    if not isinstance(load_guid, str):
        raise ValueError(f"Loan GUID is not a string, got {type(load_guid)}")
    return load_guid


@retry(stop=stop_after_attempt(3), reraise=True, wait=wait_fixed(5))
def get_all_retrieve_documents(
    access_token: str, api_server: str, loan_guid: str
) -> list[dict[str, Any]]:

    url = f"{api_server}/encompass/v3/loans/{loan_guid}/documents?includeRemoved=false&requireActiveAttachments=false"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": CONTENT_TYPE_JSON,
    }

    json_dict = None
    try:
        response = requests.request("GET", url, headers=headers)
        json_dict = json.loads(response.text)
    except Exception as e:
        logger.info(f"Unable to retrieve_documents from econnect {e}")
        raise ValueError("Unable to retrieve_documents from econnect") from e

    if json_dict is None:
        raise ValueError("Response is None")
    if not isinstance(json_dict, list):
        raise ValueError(f"Expected list response, got {type(json_dict)}")
    return json_dict


@retry(
    stop=stop_after_attempt(3),
    reraise=True,
    wait=wait_fixed(5),
    retry=retry_if_not_exception_type(LoanLockConflictError),
)
def create_new_document(access_token: str, api_server: str, loan_guid: str, doc_title: str) -> str:

    url = f"{api_server}/encompass/v3/loans/{loan_guid}/documents?action=add&view=entity"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": CONTENT_TYPE_JSON,
    }

    payload = json.dumps([{"title": doc_title, "description": ""}])

    response = requests.request("PATCH", url, headers=headers, data=payload, timeout=30)

    if response.status_code == 409:
        _raise_if_lock_conflict(response, loan_guid)

    try:
        logger.info(response.text)
        json_dict = json.loads(response.text)
        document_id = json_dict[0]["id"]
    except Exception as e:
        logger.info(f"Unable to create_new_document in econnect {e}")
        raise ValueError(response.text) from e

    if not document_id:
        raise ValueError("Document ID not found in response")
    if not isinstance(document_id, str):
        raise ValueError(f"Document ID is not a string, got {type(document_id)}")
    return document_id


def _raise_if_lock_conflict(response: requests.Response, loan_guid: str) -> None:
    """Raise ``LoanLockConflictError`` only for actual 409 loan lock conflicts."""
    try:
        body = response.json()
    except Exception:
        body = {}
    details = body.get("details", response.text)
    locked_match = re.search(r"locked by another user '([^']+)'", str(details))
    if not locked_match:
        return
    raise LoanLockConflictError(
        loan_guid=loan_guid,
        locked_by=locked_match.group(1),
        raw_response=response.text,
    )


def add_efolder_description(
    access_token: str,
    api_server: str,
    loan_id: str,
    loan_guid: str,
    file_name: str,
    entity_id: str,
    description: str,
) -> None:

    if not entity_id or not description:
        logger.info(
            f"{loan_id} {file_name} {description} entity id is null not able to add Description"
        )
        return

    url = f"{api_server}/encompass/v3/loans/{loan_guid}/documents?action=update&view=entity"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": CONTENT_TYPE_JSON,
    }

    current_datetime = datetime.datetime.now(datetime.UTC)
    received_date = current_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = json.dumps(
        [{"id": entity_id, "description": description, "receivedDate": received_date}]
    )

    try:
        requests.request("PATCH", url, headers=headers, data=payload)
        logger.info(f"{loan_id} {file_name} {description} ")

    except Exception as e:
        logger.info(
            f"{loan_id} {file_name} {description} Unable to create_new_document in econnect {e}"
        )


@retry(stop=stop_after_attempt(3), reraise=True, wait=wait_fixed(5))
def get_loan_details(access_token: str, api_server: str, loan_guid: str) -> dict[str, Any]:

    url = f"{api_server}/encompass/v3/loans/{loan_guid}"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": CONTENT_TYPE_JSON,
    }

    try:
        response = requests.request("GET", url, headers=headers)
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError(f"Expected dict response, got {type(result)}")
        return result
    except Exception as e:
        logger.info(f"Unable to retrieve_documents from econnect {e}")
        raise ValueError("Unable to retrieve_documents from econnect") from e
