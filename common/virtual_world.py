"""Defines a top level container object for a virtual world."""


import math
import random
import itertools

from .state_node import StateNode
from .state_mutating_actions import Transfer, Transform
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
        self._simulation_root_node = None

        # HYPERPARAMETERS
        self._MAX_TRANSFORM_SCALAR = 0.75
        self._MAX_TRANSFER_SCALAR = 0.33
        self._RANDOM_POSSIBLE_NEXT_STATES_SCALAR = 0.25
        self._REWARD_DISCOUNT_GAMMA = 0.05
        self._TRANSFORM_SUCCESS_PROBABILITY = 0.925

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
    

    def _apply_state_quality_function(self, state: StateNode):
        """Calculate the state quality for a given StateNode.
        
        Per the description in the project write up I started with the following State Quality Function:

        Weighted sum of resource factors, normalized by the population resource, such as wRi ∗ cRi ∗ ARi /APopulation, 
        where A Ri is the amount of a resource, and c Ri is a proportionality constant (e.g., 2 units food 
        per person, 0.5 houses per person). The proportionality constant is taken from the resource weights initially
        read in during program initialization.
        
        Parameters
        --------------------
        node : StateNode
            the node whose possible Transfer child states are being determined
        
        Returns
        --------------------
        state_quality: float
            A float representing the computed state quality
        """
        state_quality = 0

        # Initial proportionality constants NOTE: These will need to eventually be reexamined
        PROPORTIONALITY_CONSTANTS = {
            'Housing' : 0.5,
        }

        # Get the resource dictionary for 'self'
        world_state_dict_for_node = state.world_state.get_world_dict()
        primary_actor_state = world_state_dict_for_node[self.primary_actor_country]
        population = primary_actor_state['Population']

        # Calculate the state quality by iterating over the resources
        for resource, amount in primary_actor_state.items():
            weight = self.resource_weights.get_weight_for_resource(resource)
            proportionality_constant = float(PROPORTIONALITY_CONSTANTS[resource]) if resource in PROPORTIONALITY_CONSTANTS.keys() else 1
            

            impact = (weight * proportionality_constant * amount) / population
            state_quality += impact

        # Set the value in the StateNode instance
        state._state_quality = state_quality

        return state_quality 
    

    def get_solution_schedules(self):
        """Get a list of solutions that have been found.

        Returns:
        solutions -- a list of simulation solutions in the form of a list of lists of Transfers and Transforms
        """
        solutions = self._schedules

        return solutions
    

    def _find_all_possible_transforms(self, current_self_state_dict, node: StateNode):
        """Find all possible Transform actions that can be taken at a given node.

        Keyword arguments:
        current_self_state_dict -- a Python dictionary of the resource state of the current 'self' country
        node -- an instance of StateNode whose possible Transform are being searched

        Returns:
        transform_list -- a list of StateNodes containing all possible Transform actions that can be taken
        """
        transform_list = []

        for t in self.transform_templates:
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

            # Append each possible scalar multiple to the list of potential child states, scaled via hyperparameter setting
            decrement = math.floor(max_scalar * self._MAX_TRANSFORM_SCALAR)

            while decrement > 0:
                # Instantiate a Transform and build the child node
                transform_t = Transform(country=self.primary_actor_country, transform=t, scalar=decrement, is_self=True)
                child_state_node = self._build_child_transform_node(parent_node=node, transform=transform_t)
                
                # Push to list of possible children states
                transform_list.append(child_state_node)

                decrement = decrement - 1

        return transform_list
    

    def _build_child_transform_node(self, parent_node: StateNode, transform: Transform):
        """Instantiate a StateNode instance representing a child state reached from a Transform action.

        Keyword arguments:
        parent_node-- an instance of StateNode that is the parent of this node
        transform -- an instance of Transform representing a the transform performed

        Returns:
        child_state_node -- the possible child StateNode instance
        """
        new_world_state = WorldState(isInitial=False, isClone=True, state_to_clone=parent_node.world_state)
        new_world_state.update_world_state_with_transform(transform)

        # Instantiate the StateNode of this potential child
        child_state_node = StateNode(parent_node.depth + 1, new_world_state, transform, is_root_node=False, parent=parent_node)

        return child_state_node
    

    def _find_all_possible_transfers(self, node:StateNode):
        """Initialize a simulation and a virtual world.
        
        Parameters
        --------------------
        node : StateNode
            The node whose possible Transfer child states are being determined
        
        Returns
        --------------------
        transfer_list: list[Transfer]
            A list of StateNodes containing all possible Transfer actions that can be taken
        """
        transfer_list = []

        world_state = node.world_state.get_world_dict()
        countries = node.world_state.get_countries()
        resources = node.world_state.get_resources()
        
        # Find all permutations of countries and resources
        country_pairs = [p for p in itertools.permutations(countries, 2)]

        # Filter out transfers that don't involve the primary actor 'self'
        country_pairs = list(filter(lambda cp: cp[0] == self.primary_actor_country or cp[1] == self.primary_actor_country, country_pairs)) 

        # Loop over each country pair and resource pair to find possible transfers
        for cp in country_pairs:
            giver = cp[0]
            receiver = cp[1]
            from_self = giver == self.primary_actor_country
            to_self = receiver == self.primary_actor_country

            for r in resources:
                total_available = world_state[giver][r]

                # Scale total_available with the hyperparameter setting
                scaled_total_available = math.floor(total_available * self._MAX_TRANSFER_SCALAR)

                # Determine all possible possible Transfers
                while scaled_total_available > 0:
                    transfer_t = Transfer(from_country=giver, 
                                          to_country=receiver, 
                                          resource_name=r, 
                                          amount=scaled_total_available,
                                          from_is_self=from_self,
                                          to_is_self=to_self)
                    
                    # Build a new child StateNode for this Transfer
                    child_state_node = self._build_child_transfer_node(parent_node=node, transfer=transfer_t)

                    # Push to list of possible children states
                    transfer_list.append(child_state_node)

                    scaled_total_available  = scaled_total_available  - 1

        return transfer_list
    

    def _build_child_transfer_node(self, parent_node: StateNode, transfer: Transfer):
        """Instantiate a StateNode instance representing a child state reached from a Transfer action.

        Parameters
        --------------------
        parent_node : StateNode
            an instance of StateNode that is the parent of this node
        transfer : Transfer
            an instance of Transfer representing a the transfer performed

        Returns
        --------------------
        child_state_node : StateNode 
            the possible child StateNode instance
        """
        new_world_state = WorldState(isInitial=False, isClone=True, state_to_clone=parent_node.world_state)
        new_world_state.update_world_state_with_transfer(transfer)

        # Instantiate the StateNode of this potential child
        child_state_node = StateNode(parent_node.depth + 1, new_world_state, transfer, is_root_node=False, parent=parent_node)

        return child_state_node
    

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

        # Find all possible child state_nodes that can arise from Transform actions
        transform_child_states = self._find_all_possible_transforms(current_self_state_dict=current_self_state_dict, node=node)
        possible_child_states.extend(transform_child_states)

        # Find all possible child state_nodes that can arise from Transfer actions
        transfer_child_states = self._find_all_possible_transfers(node=node)
        possible_child_states.extend(transfer_child_states)

        node.set_child_states(possible_child_states)

        return possible_child_states
    

    def _calculate_undiscounted_reward(self, state_node: StateNode):
        """Calculate the undiscounted reward of a StateNode.

        Parameters
        --------------------
        state_node : StateNode
            The StateNode to calculate the undiscounted reward  from

        Returns
        --------------------
        undiscounted_reward : float 
            The undiscounted reward of this StateNode
        """
        undiscounted_reward = state_node._state_quality - state_node.parent._state_quality 
        state_node._undiscounted_reward = undiscounted_reward

        return undiscounted_reward
    

    def _calculate_discounted_reward(self, state_node: StateNode):
        """Calculate the discounted reward of a StateNode.

        Parameters
        --------------------
        state_node : StateNode
            The StateNode to calculate the discounted reward  from

        Returns
        --------------------
        discounted_reward : float 
            The discounted reward of this StateNode
        """
        discounted_reward = self._REWARD_DISCOUNT_GAMMA * state_node._undiscounted_reward
        state_node._discounted_reward = discounted_reward

        return discounted_reward
    

    def _calculate_expected_utility_of_transform_state_node(self, state_node: StateNode):
        """Calculate the expected utility for a Transform type StateNode.

        Parameters
        --------------------
        state_node : StateNode
            The StateNode to calculate the expected utility for

        Returns
        --------------------
        expected_utility: float 
            The expected utility of this StateNode
        """
        expected_utility = state_node._discounted_reward * self._TRANSFORM_SUCCESS_PROBABILITY
        state_node._expected_utility= expected_utility

        return expected_utility
    

    def _calculate_expected_utility_of_transfer_state_node(self, state_node: StateNode):
        """Calculate the expected utility for a Transfer type StateNode.

        Parameters
        --------------------
        state_node : StateNode
            The StateNode to calculate the expected utility for

        Returns
        --------------------
        expected_utility: float 
            The expected utility of this StateNode
        """
        expected_utility = state_node._discounted_reward * self._TRANSFORM_SUCCESS_PROBABILITY
        state_node._expected_utility= expected_utility

        return expected_utility
    

    def _calculate_expected_utility(self, state_node: StateNode):
        """Calculate the expected utility of a StateNode.

        Parameters
        --------------------
        state_node : StateNode
            The StateNode to calculate the expected utility from
        """
        # Get the state quality of each child state
        self._apply_state_quality_function(state_node)

        # Calculate the undiscounted reward of each child state
        self._calculate_undiscounted_reward(state_node)

        # Calculate the discounted reward of each child state
        self._calculate_discounted_reward(state_node)

        # Finally, calculate the expected utility
        if isinstance(state_node.action, Transfer):
            self._calculate_expected_utility_of_transfer_state_node(state_node)
        elif isinstance(state_node.action, Transform):
            self._calculate_expected_utility_of_transform_state_node(state_node)
        else:
            print("This is a something else....")

        return state_node._expected_utility

    
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
        
        self._simulation_root_node = root
        
        """
            Search for schedule solutions until either the target number of schedules
            is found or the user halts execution with Ctrl-c.
        """
        try:
            node = root

            # Calculate Expected Utility for starting state
            self._apply_state_quality_function(root)

            #while num_schedules_found < self.TARGET_NUMBER_SCHEDULES:
            #    continue
            # Find all possible child states that can be entered
            print("Finding possible child states")
            self._find_possible_child_states_for_node(node)

            # Take a random sample of these states to explore
            states_to_explore = node.get_child_states()
            print(f"Total children -> {len(states_to_explore)}")

            # Calculate the expected utility of each state
            for state in states_to_explore:
                self._calculate_expected_utility(state_node=state)





        except KeyboardInterrupt:
            # User interrupt the program with ctrl+c
            print("User has halted simulation execution early.")

        return
