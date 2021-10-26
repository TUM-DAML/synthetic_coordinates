import gzip
import json
import binascii
from typing import Any, Iterable
from collections import OrderedDict


def is_gzipped(path: str) -> bool:
    with open(path, "rb") as f:
        return binascii.hexlify(f.read(2)) == b"1f8b"


def read_binary(path: str) -> bytes:
    if is_gzipped(path):
        with gzip.open(path) as f:
            return f.read()
    else:
        with open(path, "rb") as f:
            return f.read()


def read_text(path: str) -> str:
    return read_binary(path).decode("utf-8")


def read_jsonl(path: str) -> Iterable[Any]:
    """
    Parse JSONL files. See http://jsonlines.org/ for more.
    :param error_handling: a callable that receives the original line and the exception object and takes
            over how parse error handling should happen.
    :return: a iterator of the parsed objects of each line.
    """
    for line in read_text(path).splitlines():
        yield json.loads(line, object_pairs_hook=OrderedDict)
