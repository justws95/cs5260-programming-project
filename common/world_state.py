"""Defines a WorldState object that models the world at a given state."""


from __future__ import annotations
from .exceptions import IllegalInitialWorldStateError, IllegalCloneWorldStateError
from .state_mutating_actions import Transfer, Transform



class WorldState:
    """Model a simulated world at a given state."""

    def __init__(self, isInitial=False, world_state_df=None, isClone=False, state_to_clone:'WorldState' =None):
        """Initialize a WorldState instance."""
        if isInitial:
            if world_state_df is None:
                raise IllegalInitialWorldStateError
            
            self._instantiate_from_state_df(world_state_df)
        elif isClone:
            if state_to_clone is None:
                raise IllegalCloneWorldStateError
            
            self._countries = list(state_to_clone.get_countries())
            self._resources = list(state_to_clone.get_resources())
            self._world_dict = dict(state_to_clone.get_world_dict())
        else:
            self._countries = []
            self._resources = []
            self._world_dict = {}

        return
    
    
    def __str__(self):
        """Define the string representation of this virtual world.

        Returns:
        STR -- the string representation of this class instance
        """
        STR = ""

        for c in self._countries:
            STR += f"{c}\n"
            STR += "-"*20 + "\n"

            for r in self._resources:
                STR += f"  {r}: {self._world_dict[c][r]}\n"
            
            STR += "\n"


        return STR
    

    def _instantiate_from_state_df(self, world_state_df):
        """Create an initial world state from a input pandas.DataFrame.

        Keyword arguments:
        world_state_df -- pandas.DataFrame representing the initial world state
        """
        # Set the resources
        resource_list = []

        for col in world_state_df.columns:
            if col == "Country":
                continue
            resource_list.append(col)

        self.set_resources(resource_list=resource_list)

        # Set the names of the countries
        country_list = list(world_state_df['Country'])
        self.set_countries(country_list=country_list)

        # Set up the world dict mapping
        world_dict = {}

        for country in self.get_countries():
            world_dict[country] = {}

        for row in range(len(world_state_df)):
            country = world_state_df.at[row, 'Country']

            for resource in self.get_resources():
                resource_amount = world_state_df.at[row, resource]
                world_dict[country][resource] = int(resource_amount)

        self.set_world_dict(world_dict=world_dict)

        return
    
    
    def _validate_load_from_files(self):
        """Validate that resources defined in resource file match those in state file. Throw Error if otherwise."""
        pass

    
    def get_countries(self):
        """Get a list of countries in this world.

        Returns:
        countries -- a list of all the countries in this world
        """
        return list(self._countries)
    

    def set_countries(self, country_list):
        """Set the countries for this world state.

        Keyword arguments:
        country_list -- a list of all the countries for this world state
        """
        self._countries = list(country_list)

        return
    

    def get_resources(self):
        """Get a list of resources in this world.

        Returns:
        resources -- a list of all the resources in this world
        """
        return list(self._resources)
    

    def set_resources(self, resource_list):
        """Set the resources for this world state.

        Keyword arguments:
        resource_list -- a list of all the resources for this world state
        """
        self._resources = list(resource_list)

        return
    
    def get_world_dict(self):
        """Get a Python dictionary representing the world state.

        Returns:
        world_dict -- a Python dictionary representing the world state
        """
        return dict(self._world_dict)
    

    def set_world_dict(self, world_dict):
        """Set a Python dictionary representing the world state.

        Keyword arguments:
        world_dict -- a Python dictionary representing the world state
        """
        self._world_dict = dict(world_dict)

        return
    
    def update_world_state_with_transform(self, transform: Transform):
        """Update the current world state to reflect a Transform action.

        Keyword arguments:
        transform -- an instance of Transform representing a transform that changes the world state
        """
        actor = transform.country
        scalar = transform.scalar
        inputs = transform.transform.get_inputs_tuples_list()
        outputs = transform.transform.get_outputs_tuples_list()

        # Get the dictionary of resources for the relevant actor
        actor_dict = dict(self._world_dict[actor])
        # Update by subtracting the scalar multiple of the inputs
        for key, val in inputs:
            actor_dict[key] = int(actor_dict[key]) - (int(val) * scalar)

        # Update by adding the scalar multiple of the outputs
        for key, val in outputs:
            actor_dict[key] = int(actor_dict[key]) + (int(val) * scalar)
            
        # Update the world state in the mapping
        self._world_dict[actor] = dict(actor_dict)

        return
    

    def update_world_state_with_transfer(self, transfer: Transfer):
        """Update the current world state to reflect a Transfer action.

        Keyword arguments:
        transfer -- an instance of Transfer representing a transform that changes the world state
        """
        giver = transfer.from_country
        receiver = transfer.to_country
        resource = transfer.resource
        amount = transfer.amount

        new_world_state = dict(self._world_dict)

        new_world_state[giver][resource] -= amount
        new_world_state[receiver][resource] += amount

        self.set_world_dict(new_world_state)

        return
