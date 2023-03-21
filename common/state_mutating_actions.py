"""Defines a set of classes and utilities for describing actions which mutate world state."""


from enum import Enum
from .transformation_template import TransformTemplate



class ActionType(Enum):
    """Enum denoting the types of transactions available."""

    TRANSFORM = "TRANSFORM"
    TRANSFER = "TRANSFER"


class Transfer:
    """Models a transfer of resources between 2 countries."""

    def __init__(self, from_country, to_country, resource_name, amount):
        """Initialize an instance of a Transfer."""
        self.from_country = from_country
        self.to_country = to_country
        self.resource = resource_name
        self.amount = amount

        return
    

    def __repr__(self):
        """Define the representation of this transfer.

        Returns:
        REPR -- the string representation of this class instance
        """
        REPR = f"(TRANSFER {self.from_country} {self.to_country} (({self.resource} {self.amount})))"

        return REPR
    

    def __str__(self):
        """Define the string representation of this transfer. Useful while writing schedule.

        Returns:
        STR -- the string representation of this class instance
        """
        STR = f"(TRANSFER {self.from_country} {self.to_country} (({self.resource} {self.amount})))"

        return STR
    

class Transform:
    """Models a transform of resources into other resources."""

    def __init__(self, country, transform: TransformTemplate, scalar=1):
        """Initialize an instance of a Transform."""
        self.country = country
        self.transform = transform
        self.scalar = scalar

        return
    

    def __repr__(self):
        """Define the representation of this transform.

        Returns:
        REPR -- the string representation of this class instance
        """
        REPR = f"(TRANSFORM {self.country} (INPUTS"

        inputs = self.transform.get_inputs_tuples_list()

        for i in inputs:
            REPR += f" ({i[0]} {str(int(i[1]) * self.scalar)})"

        REPR += ") (OUTPUTS"
        
        outputs = self.transform.get_outputs_tuples_list()

        for o in outputs:
            REPR += f" ({o[0]} {str(int(o[1]) * self.scalar)})"

        REPR += "))"

        return REPR
    

    def __str__(self):
        """Define the string representation of this transfer. Useful while writing schedule.

        Returns:
        STR -- the string representation of this class instance
        """
        STR = f"(TRANSFORM {self.country} (INPUTS"

        inputs = self.transform.get_inputs_tuples_list()

        for i in inputs:
            STR  += f" ({i[0]} {str(int(i[1]) * self.scalar)})"

        STR += ") (OUTPUTS"
        
        outputs = self.transform.get_outputs_tuples_list()

        for o in outputs:
            STR  += f" ({o[0]} {str(int(o[1]) * self.scalar)})"

        STR += "))"

        return STR
    