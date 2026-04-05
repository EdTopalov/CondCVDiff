import os
from pathlib import Path


def get_project_path():
    """
    Reads path to the root of project
    Returns:
        path(str): path to root of project
    """
    return Path(__file__).parent.parent.parent