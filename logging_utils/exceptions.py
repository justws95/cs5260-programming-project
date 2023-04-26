"""Defines exceptions that can be raised from methods of the logging_utils module."""


class CannotCreateLogFileError(Exception):
    """Exception raised when the specified logfile cannot be created."""

    def __init__(self, log_file_path):
        """Initialize a CannotCreateLogFileError instance.
        
        Parameters
        --------------------
        log_file_path: str
            The file path of the output log file.
        """
        message = f"Could not create the log file at path {log_file_path}."
        super().__init__(message)

        return
    

class NoSimulationLoggerFoundError(Exception):
    """Exception raised when an instance of SimulationLogger cannot be retrieved."""

    def __init__(self, log_file_path):
        """Initialize a NoSimulationLoggerFound instance."""
        message = f"No existing instance of SimulationLogger was found. Instantiate a new instance before calling get_logger method."
        super().__init__(message)

        return
    