"""Defines some utility functions to assist in initial world loading."""


import errno
import os
import pandas as pd

from common import ResourceWeights, WorldState


def _exists(path):
    """Test whether a path exists.  Returns False for broken symbolic links.

    Code copied from stack overflow -> 
    https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists-without-exceptions

    Keyword arguments:
    path -- the path to the file whose existence is being checked
    """
    try:
        st = os.stat(path)
    except os.error:
        return False
    
    return True


def _convert_world_state_df_to_WorldState(world_state_df):
    """Convert an initial world state pandas.DataFrame into a WorldState instance.

    Keyword arguments:
    world_state_df -- pandas.DataFrame representing the initial world state

    Returns:
    initial_world -- an instance of WorldState representing the initial world state
    """
    initial_world = WorldState(isInitial=True, world_state_df=world_state_df)

    return initial_world


def _convert_resources_df_to_ResourceWeights(resource_df):
    """Convert a resources pandas.DataFrame into a ResourceWeights instance.

    Keyword arguments:
    resource_df -- pandas.DataFrame representing the relative resource weights.

    Returns:
    resource_weights -- an instance of ResourceWeights representing the relative resource weights.
    """
    resource_weights = ResourceWeights(weights_df=resource_df)

    return resource_weights


def load_initial_state_file(path):
    """Load the initial world state file for countries and resources.

    Keyword arguments:
    path -- the file path to the initial world state file

    Returns:
    initial_world -- an instance of WorldState representing the initial world state
    """
    if not _exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    
    world_state_df = pd.read_csv(path)

    initial_world = _convert_world_state_df_to_WorldState(world_state_df)
    
    return initial_world
    

def load_resources_file(path):
    """Load the resource names, weights, and factor definitions.

    Keyword arguments:
    path -- the file path to the resource file
    """
    if not _exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    
    resource_df = pd.read_csv(path)

    resources = _convert_resources_df_to_ResourceWeights(resource_df)
    
    return resources
