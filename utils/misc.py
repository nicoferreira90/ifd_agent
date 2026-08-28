import glob
import json
import os
from typing import Any

import pypdf
from af.tools import logger


def get_file_metadata(file_path: str) -> tuple[str | None, str | None]:
    filename_list = file_path.split("_")

    loan_id = None
    file_name = None

    if len(filename_list) > 1:
        loan_id = filename_list[0]
        file_name = filename_list[1].replace(".pdf", "")

    return loan_id, file_name


def get_pdf_from_local_storage(file_path: str) -> tuple[bytes | None, int]:
    file_content = None
    file_size = 0

    with open(file_path, mode="rb") as file:
        file_content = file.read()

    try:
        file_size = os.path.getsize(file_path)
    except Exception as e:
        logger.debug(f"Failed to get file size for {file_path}: {e}")

    return file_content, file_size


def is_valid_pdf(file_obj: Any) -> bool:
    try:
        reader = pypdf.PdfReader(file_obj)
        if reader.is_encrypted:
            logger.debug("PDF is encrypted and cannot be validated as readable")
            return False
        return len(reader.pages) > 0
    except Exception as e:
        logger.debug(f"Failed to read PDF: {e}")
        return False


def load_efolder_mapping(file_path: str) -> dict[str, Any]:
    with open(file_path) as f:
        json_dict = json.load(f)

    if not isinstance(json_dict, dict):
        raise ValueError(f"Expected dict from efolder mapping file, got {type(json_dict)}")
    return json_dict


def find_efolder_mapping_id(
    econnect_document_list: list[dict[str, Any]], efolder_file_name: str
) -> str | None:
    document_id = None
    efolder_file_name = efolder_file_name.strip().lower()

    for doc_dict in econnect_document_list:
        if efolder_file_name == doc_dict["title"].strip().lower():
            document_id = doc_dict["id"]
            break

    return document_id


def cleanup_pdf_storage(pdf_path: str | None = None) -> dict[str, Any]:
    """Clean up PDF files from storage directories to prevent accumulation in containers."""
    result: dict[str, Any] = {
        "success": True,
        "files_deleted": 0,
        "total_size_freed": 0,
        "errors": [],
    }

    try:
        storage_dirs = []
        if os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
            storage_dirs.append("/tmp/pdf_storage")
        else:
            storage_dirs.append("pdf_storage")
            if os.path.exists("/tmp/pdf_storage"):
                storage_dirs.append("/tmp/pdf_storage")

        if pdf_path:
            if os.path.exists(pdf_path):
                try:
                    file_size = os.path.getsize(pdf_path)
                    os.remove(pdf_path)
                    result["files_deleted"] = 1
                    result["total_size_freed"] = file_size
                    logger.info(f"Cleaned up PDF file: {pdf_path} ({file_size:,} bytes)")
                except Exception as e:
                    error_msg = f"Failed to delete {pdf_path}: {str(e)}"
                    result["errors"].append(error_msg)
                    result["success"] = False
                    logger.warning(error_msg)
            else:
                logger.debug(f"PDF file not found (may have been already cleaned): {pdf_path}")
        else:
            for storage_dir in storage_dirs:
                if not os.path.exists(storage_dir):
                    continue

                pdf_pattern = os.path.join(storage_dir, "*.pdf")
                pdf_files = glob.glob(pdf_pattern)

                for pdf_file in pdf_files:
                    try:
                        file_size = os.path.getsize(pdf_file)
                        os.remove(pdf_file)
                        result["files_deleted"] += 1
                        result["total_size_freed"] += file_size
                        logger.debug(f"Deleted: {pdf_file} ({file_size:,} bytes)")
                    except Exception as e:
                        error_msg = f"Failed to delete {pdf_file}: {str(e)}"
                        result["errors"].append(error_msg)
                        result["success"] = False
                        logger.warning(error_msg)

        if result["files_deleted"] > 0:
            size_mb = result["total_size_freed"] / (1024 * 1024)
            logger.info(
                f"PDF storage cleanup completed: {result['files_deleted']} file(s) removed, "
                f"{size_mb:.2f} MB freed"
            )
        else:
            logger.debug("PDF storage cleanup: No files to clean")

    except Exception as e:
        error_msg = f"Error during PDF cleanup: {str(e)}"
        result["errors"].append(error_msg)
        result["success"] = False
        logger.error(error_msg)

    return result
