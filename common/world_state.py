"""Defines a WorldState object that models the world at a given state."""


from .exceptions import IllegalInitialWorldStateError



class WorldState:
    """Model a simulated world at a given state."""

    def __init__(self, isInitial=False, world_state_df=None):
        """Initialize a WorldState instance."""
        if isInitial:
            if world_state_df is None:
                raise IllegalInitialWorldStateError
            
            self._instantiate_from_state_df(world_state_df)
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
                world_dict[country][resource] = resource_amount

        self.set_world_dict(world_dict=world_dict)

        return

    
    def get_countries(self):
        """Get a list of countries in this world.

        Returns:
        countries -- a list of all the countries in this world
        """
        return self._countries
    

    def set_countries(self, country_list):
        """Set the countries for this world state.

        Keyword arguments:
        country_list -- a list of all the countries for this world state
        """
        self._countries = country_list

        return
    

    def get_resources(self):
        """Get a list of resources in this world.

        Returns:
        resources -- a list of all the resources in this world
        """
        return self._resources
    

    def set_resources(self, resource_list):
        """Set the resources for this world state.

        Keyword arguments:
        resource_list -- a list of all the resources for this world state
        """
        self._resources = resource_list

        return
    
    def get_world_dict(self):
        """Get a Python dictionary representing the world state.

        Returns:
        world_dict -- a Python dictionary representing the world state
        """
        return self._world_dict
    

    def set_world_dict(self, world_dict):
        """Set a Python dictionary representing the world state.

        Keyword arguments:
        world_dict -- a Python dictionary representing the world state
        """
        self._world_dict = world_dict

        return
