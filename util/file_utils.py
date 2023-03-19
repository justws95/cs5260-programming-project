"""Defines utility functions used in common by loaders and parsers."""


import os



def exists(path):
    """Test whether a path exists.  Returns False for broken symbolic links.

    Code copied from stack overflow -> 
    https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists-without-exceptions

    Keyword arguments:
    path -- the path to the file whose existence is being checked

    Returns:
    does_exist -- boolean value result
    """
    does_exist = True

    try:
        st = os.stat(path)
    except os.error:
        does_exist = False

    return does_exist


def file_is_empty(path):
    """Test whether a file at the specified path is empty.

    Code copied from stack overflow -> 
    https://stackoverflow.com/questions/2507808/how-to-check-whether-a-file-is-empty-or-not

    Keyword arguments:
    path -- the path to the file that is being checked

    Returns:
    is_empty -- boolean value result
    """
    is_empty = os.stat(path).st_size==0

    return is_empty
 