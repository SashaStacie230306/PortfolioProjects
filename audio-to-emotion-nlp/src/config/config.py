"""Configuration module for the transcription service."""

import os
from typing import List


def get_api_key() -> str:
    """Retrieve the API key for the transcription service.

    Returns:
        str: The API key as a string.
    """
    return "fd9b0f0712de418f921d3544afd50a94"


def get_headers() -> dict[str, str]:
    """Generate the request headers including the authorization token.

    Returns:
        dict: Dictionary containing the authorization header.
    """
    return {"authorization": get_api_key()}


def get_first_mp3_from_folder(folder_path: str) -> str:
    """Return the path of the first MP3 file found in the specified folder.

    Args:
        folder_path (str): Path to the folder to search in.

    Returns:
        str: Path to the first MP3 file found.

    Raises:
        FileNotFoundError: If no MP3 file is found in the given folder.
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".mp3"):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No MP3 files found in the folder: {folder_path}")


def get_file_paths() -> List[str]:
    """Get a list of MP3 file paths to be processed.

    Searches the 'src/data/mp3' folder and returns the first MP3 file found.

    Returns:
        List[str]: A list containing the normalized path to the first MP3 file
        found. Returns an empty list if no MP3 file is found.
    """
    folder_path = os.path.join("src", "data", "mp3")
    try:
        first_mp3 = get_first_mp3_from_folder(folder_path)
        print(f"Found MP3 file: {first_mp3}")
        return [os.path.normpath(first_mp3)]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []
