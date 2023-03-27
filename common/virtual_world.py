"""Defines a top level container object for a virtual world."""


import math
import heapq
import random
import itertools

from copy import deepcopy

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
        self.valid_resource_names = resource_weights.get_resource_names()
        self.transform_templates = transform_templates
        self.primary_actor_country = self_country_name
        
        self.TARGET_NUMBER_SCHEDULES = target_num_schedules
        self.DEPTH_BOUND = depth_bound
        self.MAX_FRONTIER_SIZE = frontier_size_limit

        self._schedules = []
        self._simulation_root_node = None
        self._simulation_root_node_quality = None

        # Cache transforms
        self._transform_cache = {}

        # HYPERPARAMETERS
        self._MIN_TRANSFORM_SCALAR = 0.15
        self._MAX_TRANSFORM_SCALAR = 0.75
        self._MAX_TRANSFER_SCALAR = 0.50
        self._MIN_TRANSFER_SCALAR = 0.05
        self._RANDOM_POSSIBLE_NEXT_STATES_SCALAR = 1.0
        self._REWARD_DISCOUNT_GAMMA = 0.05
        self._TRANSFORM_SUCCESS_PROBABILITY = 0.925
        self._SCHEDULE_FAILURE_REWARD = -1.5
        self._LOGISTIC_FUNCTION_L = 1
        self._LOGISTIC_FUNCTION_X_NOT = 0
        self._LOGISTIC_FUNCTION_K = 1
        self._HOUSING_DEVIATION_SCALAR = 0.1

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
        state : StateNode
            The state whose quality is being determined
        
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

            # Scale housing to target 4 people per house
            if resource == 'Housing':
                population_to_housing_ratio = amount / population
                distance_from_target_ratio = abs(4 - population_to_housing_ratio)

                # Scale housing to this ratio
                impact -= (distance_from_target_ratio * self._HOUSING_DEVIATION_SCALAR)

            state_quality += impact

        # Set the value in the StateNode instance
        state._state_quality = state_quality

        return state_quality


    # NOTE: This really could be done more optimally and needs a refactor, ignoring DRY principle for now...
    def _apply_state_quality_function_to_dict(self, state: dict):
        """Calculate the state quality for a given dict.
        
        Per the description in the project write up I started with the following State Quality Function:

        Weighted sum of resource factors, normalized by the population resource, such as wRi ∗ cRi ∗ ARi /APopulation, 
        where A Ri is the amount of a resource, and c Ri is a proportionality constant (e.g., 2 units food 
        per person, 0.5 houses per person). The proportionality constant is taken from the resource weights initially
        read in during program initialization.
        
        Parameters
        --------------------
        state : dict
            The dict whose possible Transfer child states are being determined
        
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

        # Get the state for the relevant actor
        actor_state = state
        population = actor_state['Population']

        # Calculate the state quality by iterating over the resources
        for resource, amount in actor_state.items():
            weight = self.resource_weights.get_weight_for_resource(resource)
            proportionality_constant = float(PROPORTIONALITY_CONSTANTS[resource]) if resource in PROPORTIONALITY_CONSTANTS.keys() else 1
            impact = (weight * proportionality_constant * amount) / population

            # Scale housing to target 4 people per house
            if resource == 'Housing':
                population_to_housing_ratio = amount / population
                distance_from_target_ratio = abs(4 - population_to_housing_ratio)

                # Scale housing to this ratio
                impact -= (distance_from_target_ratio * self._HOUSING_DEVIATION_SCALAR)

            state_quality += impact

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
            # Add catch for templates that don't exist in this world
            transform_resources = []
            skip_transform = False

            for i in t.get_inputs_list():
                transform_resources.append(i)
            for o in t.get_outputs_list():
                transform_resources.append(o)

            # Check if this transform is possible for this world (i.e. that transform resources exist in weighted resources)
            for tr in transform_resources:
                if tr not in self.valid_resource_names:
                    skip_transform = True
            
            if skip_transform:
                continue
            
            # Find all possible transforms
            t_as_dict = {}

            input_tuples = t.get_inputs_tuples_list()
            input_tuples_str = "".join([str(t[0]) + str(t[1] + "_") for t in input_tuples])

            if input_tuples_str not in self._transform_cache.keys():
                for key, val in input_tuples:
                    t_as_dict[key] = val

                # Cache this to avoid duplicate work in later iterations
                self._transform_cache[input_tuples_str] = deepcopy(t_as_dict)
            else:
                t_as_dict = self._transform_cache[input_tuples_str]

            # Determine scalar multiples that are reachable and append backwards
            scalar_dict = {}

            for key, val in t_as_dict.items():
                val = int(val)
                scalar = math.floor(current_self_state_dict[key] / val)
                scalar_dict[key] = scalar
            
            # The maximum Transform scalar is the minimum resource scalar
            max_scalar = min(scalar_dict.values())

            # Calculate the maximum and minimum number of Transfers that can be performed
            max_number_transforms = math.floor(max_scalar * self._MAX_TRANSFORM_SCALAR)
            min_number_transforms = math.floor(max_scalar * self._MIN_TRANSFORM_SCALAR)

            # Append each possible scalar multiple to the list of potential child states, scaled via hyperparameter setting
            decrement = max_number_transforms

            while decrement > min_number_transforms:
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

        # Do not allow for the transfer or Population
        resources = [r for r in resources if r != 'Population']
        
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
                min_number_transfers = math.floor(total_available * self._MIN_TRANSFER_SCALAR)

                # Determine all possible possible Transfers
                while scaled_total_available > min_number_transfers:
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
        prior_world_state = deepcopy(new_world_state)

        new_world_state.update_world_state_with_transfer(transfer)

        # Instantiate the StateNode of this potential child
        child_state_node = StateNode(parent_node.depth + 1, new_world_state, transfer, is_root_node=False, parent=parent_node)

        # Set the pre-action world state for this StateNode
        child_state_node.set_pre_action_world_state(prior_world_state)

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
        undiscounted_reward = state_node._state_quality - self._simulation_root_node_quality
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
        discounted_reward = pow(self._REWARD_DISCOUNT_GAMMA, state_node.depth) * state_node._undiscounted_reward
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
    

    def _calculate_country_accepts_transfer_probability(self, discounted_reward: float):
        """Calculate the probability that a country accepts a transfer using the logistic function.

        Parameters
        --------------------
        discounted_reward : float
            The discounted reward of this transfer

        Returns
        --------------------
        prob : float 
            The probability this transfer is accepted
        """
        L = self._LOGISTIC_FUNCTION_L
        x_0 = self._LOGISTIC_FUNCTION_X_NOT
        k = self._LOGISTIC_FUNCTION_K
        e = math.e # Euler's number
        x = discounted_reward

        prob = L / (1 + pow(e, (-k * (x - x_0))))

        return prob
    

    def _calculate_expected_utility_of_transfer_state_node(self, state_node: StateNode):
        """Calculate the expected utility for a Transfer type StateNode.

        Parameters
        --------------------
        state_node : StateNode
            The StateNode to calculate the expected utility for

        Returns
        --------------------
        expected_utility : float 
            The expected utility of this StateNode
        """
        # Get the state of the relevant actors pre and post Transform
        prior_world_state = state_node.get_pre_action_world_state().get_world_dict()
        new_world_state = state_node.world_state.get_world_dict()

        giver = state_node.action.from_country 
        receiver = state_node.action.to_country

        if giver == self.primary_actor_country:
            alternate_actor = receiver
        elif receiver == self.primary_actor_country:
            alternate_actor = giver
        else:
            print("THIS STATE SHOULD NEVER BE REACHED!!!")

        alternate_actor_root_state = self._simulation_root_node.world_state.get_world_dict()[alternate_actor]
        alternate_actor_new_world_state = new_world_state[alternate_actor]

        # Calculate the discounted reward for the non-self country
        root_quality = self._apply_state_quality_function_to_dict(alternate_actor_root_state)
        post_quality = self._apply_state_quality_function_to_dict(alternate_actor_new_world_state)

        # Calculate the discounted reward
        alternate_actor_undiscounted_reward = post_quality - root_quality
        alternate_actor_discounted_reward = pow(self._REWARD_DISCOUNT_GAMMA, state_node.depth) * alternate_actor_undiscounted_reward

        # Calculate the probability that each country accepts the transfer
        alternate_actor_accepts = self._calculate_country_accepts_transfer_probability(alternate_actor_discounted_reward)
        primary_actor_accepts = self._calculate_country_accepts_transfer_probability(state_node._discounted_reward)

        # Calculate the probability that the schedule will succeed
        schedule_success_prob = alternate_actor_accepts * primary_actor_accepts

        # Finally, calculate the expected utility of the transfer
        expected_utility = (schedule_success_prob * state_node._discounted_reward) + ((1 - schedule_success_prob) * self._SCHEDULE_FAILURE_REWARD)

        state_node._expected_utility = expected_utility

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
            print("This case should never have been reached!")

        return state_node._expected_utility
    

    def _get_schedule_from_state_node_and_parents(self, node: StateNode):
        """Traverse back up the search tree to get the schedule to arrive at this state.

        Parameters
        --------------------
        node : StateNode
            The StateNode whose schedule is being retrieved

        Returns
        --------------------
        schedule : list[StateNodes]
            A list of StateNodes representing the schedule
        """
        schedule = []

        while node != self._simulation_root_node:
            schedule.append(node)
            node = node.parent

        # Reverse the list to appear in chronological order
        schedule.reverse()

        return schedule


    def _recursively_search_for_schedules(self, node: StateNode, frontier: list):
        """Recursively find TARGET_NUMBER_SCHEDULES solution schedules.

        Parameters
        --------------------
        node : StateNode
            The StateNode to be recursively searched
        frontier : list[StateNodes]
            Priority queue of StateNodes comprising the search frontier

        Returns
        --------------------
        schedule : list[StateNodes]
            A list of StateNodes representing the schedule
        """
        print(f"Current Depth in Search Tree: {node.depth}")
        schedule = []

        # Check if base case (i.e. Targeted Depth) has been reached
        if node.depth >= self.DEPTH_BOUND:
            schedule = self._get_schedule_from_state_node_and_parents(node)
            return schedule

        self._find_possible_child_states_for_node(node)

        # Take a random sample of these states to explore
        states_to_explore = node.get_child_states()
        sample_quantifier = math.floor(len(states_to_explore) * self._RANDOM_POSSIBLE_NEXT_STATES_SCALAR)
        states_to_explore = random.sample(states_to_explore, sample_quantifier)

        # Calculate the expected utility of each state
        for state in states_to_explore:
            self._calculate_expected_utility(state_node=state)

        # Push the scored nodes into the priority queue
        while len(states_to_explore) > 0:
            if len(frontier) < self.MAX_FRONTIER_SIZE:
                heapq.heappush(frontier, (1 / state.get_expected_utility(), states_to_explore.pop()))
            else:
                heapq.heappushpop(frontier, (1 / state.get_expected_utility(), states_to_explore.pop()))

        # Free up some memory
        node.set_child_states([])

        """
        for state in states_to_explore:
            if len(frontier) < self.MAX_FRONTIER_SIZE:
                heapq.heappush(frontier, (1 / state.get_expected_utility(), state))
            else:
                heapq.heappushpop(frontier, (1 / state.get_expected_utility(), state))
        """

        # Greedily pop the largest expected utility state from the frontier
        best_next_state = heapq.heappop(frontier)
        best_next_state_node = best_next_state[1]

        # Recursively search
        schedule = self._recursively_search_for_schedules(best_next_state_node, frontier=frontier)

        return schedule
    
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

        # Search for solution schedules
        try:
            # Calculate Expected Utility for starting state
            self._apply_state_quality_function(root)
            self._simulation_root_node_quality = root._state_quality

            while len(self._schedules) < self.TARGET_NUMBER_SCHEDULES:
                frontier = []
                schedule = self._recursively_search_for_schedules(root, frontier=frontier)
                self._schedules.append(schedule)
                print(f"Schedule has been found {len(self._schedules)} of {self.TARGET_NUMBER_SCHEDULES}")
                frontier.clear()
        except KeyboardInterrupt:
            # User interrupt the program with ctrl+c
            print("User has halted simulation execution early.")

        return
    
    def get_simulation_schedules(self):
        """Fetch the schedules that were found during the simulation.

        Returns
        --------------------
        schedules : list[StateNodes]
            A list of StateNodes representing the schedules the simulation found
        """
        schedules = list(self._schedules)

        return schedules
