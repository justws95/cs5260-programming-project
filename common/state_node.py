"""Defines a node for the search tree. Contains a world state at a given depth."""


class StateNode:
    """Model a node in the search tree traversal. Contains a WorldState instance."""

    def __init__(self, depth, world_state, action, is_root_node=False, parent=None):
        """Initialize a StateNode instance."""
        self.depth = depth
        self.world_state = world_state
        self.action = action

        self._possible_child_states = []
        self._expected_utility = None

        if not is_root_node and parent is None:
            raise SyntaxError("StateNode must specify a parent node if is_root_node is False")
        
        if is_root_node and depth != 0:
            raise SyntaxError("StateNode cannot have non-zero depth when is_root_node is True")
        
        self.is_root = is_root_node
        self.parent = parent

        return
    

    def get_child_states(self):
        """Get a list of the possible child states of this node.

        Returns:
        child_states -- a list of StateNodes that are the possible child 
            states that can be reached from this node
        """
        child_states = self._possible_child_states

        return child_states
    

    def set_child_states(self, child_states):
        """Set the list of the possible child states of this node.

        Keyword arguments:
        child_states-- a Python list of StateNodes representing possible child-states of this node
        """
        self._possible_child_states = child_states

        return 
    