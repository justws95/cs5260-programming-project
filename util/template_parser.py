"""Defines some utility functions to help with loading template files.

Before writing this class, I referenced the Piazza post by John Ford 
and was strongly influenced by the approach he took. I borrowed many concepts 
and adapted them to fit my design approach, chiefly his regex for parsing the template. 
I want to give him the appropriate credit and link his post
https://piazza.com/class/lbpfjbrwi0ca3/post/23.
"""


import os

from .file_utils import exists, file_is_empty
from common import TransformTemplate, TemplateFileNotFoundError, EmptyTemplateFileError



TEMPLATE_FILE_KEYWORDS = ["TRANSFORM", "INPUTS", "OUTPUTS"]


def _read_template_file(path):
    """Read a template files contents. Really will load any files contents."""
    file_contents = None

    with open(path, mode='r') as f:
        file_contents = f.read()
    
    return file_contents



def parse_transform_template(path):
    """Top level utility function to parse a template file.

    Keyword arguments:
    path -- the file path to the template file

    Returns:
    transform -- an instance of TransformTemplate representing a transform.
    """
    # Validate the file
    if not exists(path):
        raise TemplateFileNotFoundError(path)
    
    if file_is_empty(path=path):
        raise EmptyTemplateFileError
    
    # Read the file contents
    file_contents = _read_template_file(path)

    # Get the transform name
    basename = os.path.basename(path)
    transform_name = os.path.splitext(basename)[0]

    # Instantiate a TransformTemplate instance
    transform = TransformTemplate(name=transform_name, transform_file_contents=file_contents)

    return transform
