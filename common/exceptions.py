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
    
