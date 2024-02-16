import logging
import hashlib
from pathlib import Path
from typeguard import typechecked

@typechecked
def md5(file_path: Path) -> str:
    """md5
    Calculate the md5 hash of a file.

    Parameters
    ----------
    file_path: Path
        The path to the file to be hashed.

    Returns
    -------
    str: The md5 hash as a string.
    """
    logging.debug(f"Calculating md5 of {str(file_path)}")

    with open(file_path, "rb") as file:
        file_hash = hashlib.md5(file.read()).hexdigest()

    return file_hash
