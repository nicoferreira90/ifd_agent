from pathlib import Path
from typing import Any

from af.tools import logger

from ifd_agent.encompass_assistant.exp_apis import (
    add_efolder_description,
    create_new_document,
    get_all_retrieve_documents,
    upload_attachment,
)
from ifd_agent.utils.misc import find_efolder_mapping_id, load_efolder_mapping

_EFOLDER_MAPPING_PATH = Path(__file__).parent.parent / "efolder_mapping.json"


def upload_file_into_efolder(
    user_name: str,
    api_server: str,
    access_token: str,
    loan_id: str,
    loan_guid: str,
    portal: str,
    pdf_name: str,
    pdf_obj: Any,
    pdf_size: int,
) -> str:
    """Upload pdf into Encompass eFolder"""

    dash_efolder_mapping_dict = load_efolder_mapping(str(_EFOLDER_MAPPING_PATH))
    econnect_document_list = get_all_retrieve_documents(access_token, api_server, loan_guid)
    logger.info(f"econnect_document_list length = {len(econnect_document_list)}")

    efolder_file_name = dash_efolder_mapping_dict.get(portal.upper(), None)

    if not efolder_file_name:
        logger.error(f"No efolder mapping found for {portal}")
        raise ValueError(f"No efolder mapping found for {portal}")

    efolder_file_name, *description = efolder_file_name.split("~") if efolder_file_name else [None]

    logger.info(f"efolder_file_name {efolder_file_name} description {description}")

    if not efolder_file_name:
        logger.error(f"No efolder mapping found for {portal}")
        raise ValueError(f"No efolder mapping found for {portal}")

    efolder_file_id = None

    if efolder_file_name:
        efolder_file_id = find_efolder_mapping_id(econnect_document_list, efolder_file_name)

    logger.info(f"efolder_file_id {efolder_file_id}")

    if efolder_file_name and efolder_file_id is None:

        efolder_file_id = create_new_document(
            access_token, api_server, loan_guid, efolder_file_name
        )
        logger.info(f"new efolder_file_id {efolder_file_id}")

    upload_attachment(
        access_token, api_server, loan_guid, pdf_size, pdf_name, pdf_obj, efolder_file_id, loan_id
    )
    add_efolder_description(
        access_token,
        api_server,
        loan_id,
        loan_guid,
        pdf_name,
        efolder_file_id,
        description[0] if description else "",
    )

    return f"pdf successfully uploaded into eFolder loan_id {loan_id} pdf_name {pdf_name}"
