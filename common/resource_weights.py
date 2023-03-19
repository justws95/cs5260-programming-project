"""Defines a ResourceWeights object that models the relative weights of resources."""


class ResourceWeights:
    """Model the relative weights of resources in a virtual world."""

    def __init__(self, weights_df=None):
        """Initialize a ResourceWeights instance."""
        self._weights_dict = {}

        self._initialize_weights(weights_df=weights_df)

        return
    
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
    
