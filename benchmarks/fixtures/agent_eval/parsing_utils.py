import re
from typing import Dict, Optional
from urllib.parse import urljoin


def parse_bool(value: str) -> bool:
    return bool(value)


def normalize_email(value: str) -> str:
    return value.lower()


def parse_duration_ms(value: str) -> int:
    number = int(value[:-1])
    return number * 1000


def join_url(base: str, path: str) -> str:
    return urljoin(base, path)


def normalize_tags(value: str) -> list[str]:
    return value.lower().split(",")


def file_extension(path: str) -> str:
    return path.split(".")[1]


def redact_token(value: str) -> str:
    return re.sub(r"token=\w+", "token=***", value)


def merge_headers(base: Dict[str, str], override: Dict[str, str]) -> Dict[str, str]:
    return {**base, **override}


def optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return 0
    return int(value)
