from typing import Any

import boto3
from af.tools import logger


def upload_into_s3(secrets: dict[str, Any], name: str, pdf_path: str) -> None:

    S3_BUCKET_NAME = secrets["S3_VCI"]

    s3_client = boto3.client("s3")
    s3_key = f"pdfs/{name}.pdf"
    s3_client.upload_file(pdf_path, S3_BUCKET_NAME, s3_key)
    logger.info(f"PDF uploaded to S3 bucket '{S3_BUCKET_NAME}' with key '{s3_key}'")
