"""Defines the SimulationLogger class."""


import os
import sys
import logging
import datetime

from .exceptions import CannotCreateLogFileError, NoSimulationLoggerFoundError



class SimulationLogger(object):
    """A logging manager class that simplifies capturing messages.

    The SimulationLogger class 

    Attributes
    --------------------
        log_file_path: str
            The file path of the output log file.
        logging_level : int
            The logging level that has been set. Messages capture below the set logging level are ignored.
        is_verbose : bool
            Boolean value representing if log messages should also be written to stdout/stderr 
    """

    def __new__(cls, logging_level: int=None, log_file_dir: str=None, log_file_name: str=None, is_verbose: bool=True):
        """Overload the __new__ magic method to implement the singleton pattern.

        The __new__ magic method is called whenever a new object is created. By overloading
        the method, the default behavior will no longer call the __init__ magic method.
        Instead, it will check if an instance of the SimulationLogger class already exists.
        If one does, a reference to that instance will be returned. If not, __new__ will
        call the init method to create an instance. This design enforces the singleton 
        design pattern and will only allow one instance of the class to be created.
        
        Parameters
        --------------------
        logging_level : int
            The logging level that has been set. Messages capture below the set logging level are ignored.
        log_file_dir: str
            The directory of the output log file.
        log_file_name: str
            The name of the output log file.
        is_verbose : bool
            Boolean value representing if log messages should also be written to stdout/stderr

        Returns
        --------------------
        simulation_logger : SimulationLogger
            An instance of SimulationLogger
        """
        simlog_id = "__simlog__"
        simulation_logger = cls.__dict__.get(simlog_id, None)

        if simulation_logger is not None:
            return simulation_logger
        
        simulation_logger = object.__new__(cls)
        setattr(cls, simlog_id, simulation_logger)
        simulation_logger.init(logging_level=logging_level, log_file_dir=log_file_dir, is_verbose=is_verbose)

        return simulation_logger


    def init(self, logging_level: int=None, log_file_dir: str=None, log_file_name: str=None, is_verbose: bool=True):
        """Initialize the singleton instance to set logging attributes.
        
        Parameters
        --------------------
        logging_level : int
            The logging level that has been set. Messages capture below the set logging level are ignored.
        log_file_dir: str
            The directory of the output log file.
        log_file_name: str
            The name of the output log file.
        is_verbose : bool
            Boolean value representing if log messages should also be written to stdout/stderr
        """
        if logging_level is not None:
            self.logging_level = logging_level
        else:
            self.logging_level = logging.DEBUG

        self.log_file_dir = log_file_dir
        self.log_file_name = log_file_name
        self.is_verbose = is_verbose

        self.DEFAULT_LOG_FILE_NAME = "program_log.log"

        self._init_logging_to_file()
        

        return
    

    def get_logger(self):
        """Retrieve an instance of Simulation if one exists.
        
        Raises
        --------------------
        NoSimulationLoggerFoundError
            Thrown when no instance of SimulationLogger has been instantiated.
        """
        if self.simulation_logger is not None:
            return self.simulation_logger
        
        raise NoSimulationLoggerFoundError
    

    def _init_logging_to_file(self):
        """Initialize logging to file."""
        if self.log_file_dir is None:
            time_stamp = str(datetime.datetime.now())
            output_dir = './logs/' + time_stamp.replace(" ", "_")
        
            os.mkdir(output_dir)

            self.log_file_dir = output_dir

        if self.log_file_name is None:
            self.log_file_name  = self.DEFAULT_LOG_FILE_NAME

        log_file = '/'.join([self.log_file_dir, self.log_file_name])
            
        try:
            logging.basicConfig(filename=log_file, encoding='utf-8', level=self.logging_level)
            logging.debug(f'Logging has been initiated. Writing to {log_file}.')
            self._print_to_stdout(f'Logging has been initiated. Writing to {log_file}.')
        except:
            raise CannotCreateLogFileError(log_file_path=log_file) 
        
        return
    

    def _print_to_stdout(self, message):
        """Print message to stdout."""
        print(message)

        return
    

    def _print_to_stderr(self, message):
        """Print message to stderr."""
        print(message, file=sys.stderr)

        return
    

    def debug(self, message, no_print=False):
        """Capture log at debug level.

        Parameters
        --------------------
        message : str
            The message to be capture
        no_print : boolean
            Overrides verbose setting when set to True
        """
        logging.debug(message)

        if self.is_verbose and no_print is False:
            self._print_to_stdout(message=message)


    def info(self, message, no_print=False):
        """Capture log at info level.

        Parameters
        --------------------
        message : str
            The message to be capture
        no_print : boolean
            Overrides verbose setting when set to True
        """
        logging.info(message)

        if self.is_verbose and no_print is False:
            self._print_to_stdout(message=message)


    def warning(self, message, no_print=False):
        """Capture log at warning level.

        Parameters
        --------------------
        message : str
            The message to be capture
        no_print : boolean
            Overrides verbose setting when set to True
        """
        logging.warning(message)

        if self.is_verbose and no_print is False:
            self._print_to_stderr(message=message)


    def critical(self, message, no_print=False):
        """Capture log at critical level.

        Parameters
        --------------------
        message : str
            The message to be capture
        no_print : boolean
            Overrides verbose setting when set to True
        """
        logging.critical(message)

        if self.is_verbose and no_print is False:
            self._print_to_stderr(message=message)

        return
