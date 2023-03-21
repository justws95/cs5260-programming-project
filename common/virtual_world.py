"""Defines a top level container object for a virtual world."""


import math

from sys import exit
from .state_node import StateNode
from .state_mutating_actions import Transform
from .world_state import WorldState



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

        self._schedules = []

        return
    

    def __repr__(self):
        """Define the string representation of this virtual world."""
        REPR = ""

        REPR += f"Primary Country           : {self.primary_actor_country}\n"
        REPR += f"Target Number of Schedules: {self.TARGET_NUMBER_SCHEDULES}\n"
        REPR += f"Depth Bound               : {self.DEPTH_BOUND}\n"
        REPR += f"Maximum Frontier Size     : {self.MAX_FRONTIER_SIZE}\n"

        REPR += "\n" + "*"*35 + "\n\n" + f"Initial World State:\n\n"
        REPR += str(self.initial_state)

        REPR += "*"*35 + "\n" + f"Transformation Templates:\n\n"

        for t in self.transform_templates:
            REPR += str(t)
            REPR += "\n"
        
        return REPR
    

    def __str__(self):
        """Define the string representation of this virtual world."""
        STR = ""

        STR += f"Primary Country           : {self.primary_actor_country}\n"
        STR += f"Target Number of Schedules: {self.TARGET_NUMBER_SCHEDULES}\n"
        STR += f"Depth Bound               : {self.DEPTH_BOUND}\n"
        STR += f"Maximum Frontier Size     : {self.MAX_FRONTIER_SIZE}\n"

        STR += "\n" + "*"*35 + "\n\n" + f"Initial World State:\n\n"
        STR += str(self.initial_state)

        STR += "\n" + "*"*35 + "\n\n" + f"Resource Weights:\n\n"
        STR += str(self.resource_weights)

        STR += "*"*35 + "\n" + f"Transformation Templates:\n\n"

        for t in self.transform_templates:
            STR += str(t)
            STR += "\n"
        
        return STR
    

    def get_solution_schedules(self):
        """Get a list of solutions that have been found so far.

        Returns:
        solutions -- a list of simulation solutions in the form of a list of lists of Transfers and Transforms
        """
        solutions = self._schedules

        return solutions
    

    def _build_child_transform_node(self, child_node_depth, transfer, transfer_scalar=1):
        """Instantiate a StateNode instance representing a child state reached from a Transform action.

        Keyword arguments:
        child_node_depth -- the depth of the child node in the search tree
        transform -- the transform being performed

        Returns:
        possible_child_states -- the possible states that can be reached from this node
        """


        return

    def _find_possible_child_states_for_node(self, node: StateNode):
        """Find all states that can be reached from this node via transforms or transfers.

        Keyword arguments:
        node -- the node whose possible child states are being examined

        Returns:
        possible_child_states -- the possible states that can be reached from this node
        """
        possible_child_states = []

        # Get the current world state and the state of 'self' at this node
        current_world_state_dict = node.world_state.get_world_dict()
        current_self_state_dict = current_world_state_dict[self.primary_actor_country]

        print(current_self_state_dict)

        # Find all possible Transforms that can be performed.
        for t in self.transform_templates:
            print(f"    {t.get_inputs_tuples_list()}")
            t_as_dict = {}

            for key, val in t.get_inputs_tuples_list():
                t_as_dict[key] = val

            # Determine scalar multiples that are reachable and append backwards
            scalar_dict = {}

            for key, val in t_as_dict.items():
                val = int(val)
                scalar = math.floor(current_self_state_dict[key] / val)
                scalar_dict[key] = scalar
            
            # The maximum Transform scalar is the minimum resource scalar
            max_scalar = min(scalar_dict.values())

            print(f"Max scalar -> {max_scalar}")

            # Append each possible scalar multiple to the list of potential child states
            decrement = max_scalar

            while decrement > 0:
                # Instantiate a Transform
                transform_t = Transform(self.primary_actor_country, t, decrement)

                # Update the world state
                new_world_state = WorldState(isInitial=False, isClone=True, stateToClone=node.world_state)
                new_world_state.update_world_state_with_transform(transform_t)

                # Instantiate the StateNode of this potential child
                child_state_node = StateNode(node.depth + 1, 
                                             new_world_state, 
                                             transform_t, 
                                             is_root_node=False, 
                                             parent=node)
                
                # Push to list of possible children states
                possible_child_states.append(child_state_node)

                decrement -= 1
        
        node.set_child_states(possible_child_states)

        return possible_child_states

    
    def run_simulation(self):
        """Run the simulation to find solution schedules."""
        print("Running the simulation, terminate early with Ctrl + c.")

        
        # Set up for the run of the simulation
        num_schedules_found = 0
        root = StateNode(is_root_node=True, 
                depth=0, 
                world_state=self.initial_state, 
                action=None, 
                parent=None)
        
        """
            Search for schedule solutions until either the target number of schedules
            is found or the user halts execution with Ctrl-c.
        """
        try:
            
            #while num_schedules_found < self.TARGET_NUMBER_SCHEDULES:
            #    continue
            print("Finding possible child states")
            self._find_possible_child_states_for_node(root)

        except KeyboardInterrupt:
            # User interrupt the program with ctrl+c
            print("User has halted simulation execution early.")

        return
