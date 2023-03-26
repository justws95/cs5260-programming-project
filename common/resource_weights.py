"""Defines a ResourceWeights object that models the relative weights of resources."""


class ResourceWeights:
    """Model the relative weights of resources in a virtual world."""

    def __init__(self, weights_df=None):
        """Initialize a ResourceWeights instance."""
        self._weights_dict = {}

        self._initialize_weights(weights_df=weights_df)

        return
    

    def __str__(self):
        """Define the string representation of this virtual world.

        Returns:
        STR -- the string representation of this class instance
        """
        STR = ""

        STR += "\n" + "*"*35 + "\n\n" + f"Resource Weights:\n\n"

        for key, val in self._weights_dict.items():
            STR += f"  {key}: {val}\n"

        STR += "\n"

        return STR
    

    def _initialize_weights(self, weights_df):
        """Create a weight dictionary from a pandas.DataFrame."""
        weights = {}

        for row in range(len(weights_df)):
            resource = weights_df.at[row, 'Resource']
            weight = weights_df.at[row, 'Weight']

            weights[resource] = weight

        self.set_weights_dict(weights_dict=weights)

        return


    def get_weights_dict(self):
        """Get a Python dictionary representing the relative resource weights.

        Returns:
        weights_dict -- a Python dictionary representing the relative resource weights.
        """
        return self._weights_dict
    

    def set_weights_dict(self, weights_dict):
        """Set a Python dictionary representing the relative resource weights.

        Keyword arguments:
        weights_dict -- a Python dictionary representing the relative resource weights.
        """
        self._weights_dict = weights_dict

        return
    

    def get_weight_for_resource(self, resource_name: str):
        """Retrieve the relative weight for a resource.
        
        Parameters
        --------------------
        resource_name: str
            The resource being fetched
        
        Returns
        --------------------
        weight: float
            A float representing the relative weight for this resource
        """
        weight = float(self._weights_dict[resource_name])

        return weight
    
    def get_resource_names(self):
        """Retrieve the resource names present in the weights file.
        
        Returns
        --------------------
        resources : list[str]
            A list of resource names
        """
        resources = list(self._weights_dict.keys())

        return resources
