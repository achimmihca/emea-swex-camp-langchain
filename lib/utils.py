"""
This module contains utility functions and classes for the application.

The module includes a function to retrieve the content of a file, removing any comments,
and a class that contains URLs used in the application.
"""

import os
import git


def assetpath(filename):
    """ 
    Retrieves the path of a file in the assets folder
    The filename is searched recursively in the assets folder.

    Args:
        filename (str): The name of the file to retrieve the path from.

    Returns:
        str: The path of the file.
    Raises:
        FileNotFoundError: If the file named `filename` is not found in the assets folder.
    """ 

    count = 0
    for root, _, files in os.walk("assets"):
        if filename in files:
            count += 1
            if count > 1:
                raise ValueError(f"Multiple files named {filename} found in the assets folder.")
            filename = os.path.join(root, filename)

    if count == 0:
        raise FileNotFoundError(f"File named {filename} not found in the assets folder.")

    return filename
    


def filecontent(filename):
    """
    Retrieves the content of a file.

    The filename is searched recursively in the assets folder.
    The function reads the file, removes any comments (lines starting with <!-- or //),
    and returns the remaining text.

    Args:
        filename (str): The name of the file to retrieve the content from.

    Returns:
        str: The content of the file.

    Raises:
        FileNotFoundError: If the file named `filename` is not found in the assets folder.
    """

    filename = assetpath(filename)

    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        text = ""
        for line in lines:
            if line.strip().startswith(("<!--", "//")):
                continue
            text += line
        return text
    
def get_git_info():
    """ Get the git information of the current repository."""
    try:
        repo = git.Repo("..", search_parent_directories=True)

        # Get the commit hash
        commit_hash = repo.head.commit.hexsha[0:8]

        # Check if working directory is dirty
        dirty_suffix = "-dirty" if repo.git.diff(None) or repo.git.diff(None, cached=True) else ""

        return commit_hash + dirty_suffix
    except Exception:
        return "unavailable"


    
class URLs:
    """
    A class that contains URLs used in the application.
    """
    REPORT_BUG = "https://forms.office.com/Pages/ResponsePage.aspx?id=Xn_OzF-jw0uOY7IhXn0U-aqAb4gjeKZEqD-GQGIZPclUOUZONlUyVkc3MjFHS1JaVVJVQ0wzQkI3SCQlQCN0PWcu"
