"""Defines a top level container object for a virtual world."""


class VirtualWorld:
    """Models a virtual world for the trading simulation."""

    def __init__(self, 
            initial_state, 
            resource_weights, 
            transform_templates,
            self_country_name,
            target_num_schedules,
            depth_bound,
            frontier_size_limit):
        """Initialize a VirtualWorld instance."""
        self.initial_state = initial_state
        self.resource_weights = resource_weights
        self.transform_templates = transform_templates
        self.primary_actor_country = self_country_name
        
        self.TARGET_NUMBER_SCHEDULES = target_num_schedules
        self.DEPTH_BOUND = depth_bound
        self.MAX_FRONTIER_SIZE = frontier_size_limit

        return
    