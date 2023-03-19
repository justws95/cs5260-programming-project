"""Provides some common exceptions used throughout the program."""


class IllegalInitialWorldStateError(Exception):
    """Exception raised when an initial world state is created without specifying the loaded pandas.DataFrame.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="A WorldState instance cannot be created with isInitial=True without specifying an initial state dataframe."):
        """Initialize an IllegalInitialWorldStateError instance."""
        self.message = message
        super().__init__(self.message)

        return
    

class TemplateFileNotFoundError(Exception):
    """Exception raised when a transformation template file cannot be found at the specified path.

    Attributes:
        path -- file path of the template file
    """

    def __init__(self, path):
        """Initialize a TemplateFileNotFoundError instance."""
        self.message = f"No file at {path} was found"
        super().__init__(self.message)

        return
    

class EmptyTemplateFileError(Exception):
    """Exception raised when a transformation template file is empty.

    Attributes:
        path -- file path of the template file
        message -- explanation of the error
    """

    def __init__(self, path):
        """Initialize an EmptyTemplateFileError instance."""
        self.message = f"Template file {path} was empty."
        super().__init__(self.message)

        return
    