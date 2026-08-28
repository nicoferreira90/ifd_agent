from typing import Any

from af.tools import logger

from ifd_agent.encompass_assistant.exp_apis import _get_access_token, _get_loan_guid


def connection(secrets: dict[str, Any], loan_id: str) -> tuple[str, str, str, str]:

    api_server = secrets["ENCOMPASS_API_SERVER"]
    instance_id = secrets["ENCOMPASS_INSTANCE_ID"]
    api_user_client_id = secrets["ENCOMPASS_API_USER_CLIENT_ID"]
    api_user_client_secret = secrets["ENCOMPASS_API_USER_CLIENT_SECRET"]
    encompass_username = secrets["ENCOMPASS_USERNAME"]
    encompass_password = secrets["ENCOMPASS_PASSWORD"]
    access_token = _get_access_token(
        api_server,
        instance_id,
        api_user_client_id,
        api_user_client_secret,
        encompass_username,
        encompass_password,
    )

    loan_guid = _get_loan_guid(access_token, api_server, loan_id)
    logger.info(f"loan id {loan_id} loan guid {loan_guid}")

    return loan_id, loan_guid, api_server, access_token
