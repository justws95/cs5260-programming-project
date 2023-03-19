"""Main entry point for the program."""


import os
from common import VirtualWorld
from util import load_initial_state_file, load_resources_file, parse_transform_template



def country_scheduler(your_country_name, resources_filename,
initial_state_file_name, output_schedule_file_name,
num_output_schedules, depth_bound,
frontier_max_size):
    """Top level method for running a simulation."""
    print("Creating a virtual world....\n\n\n")
    virtual_world = init_simulation(your_country_name, 
        resource_file_name=resources_filename, 
        initial_state_file_name=initial_state_file_name, 
        target_number_of_schedules=num_output_schedules, 
        depth_bound=depth_bound, frontier_max_size=frontier_max_size)
    
    print(virtual_world)


def init_simulation(primary_country_actor, 
    resource_file_name,
    initial_state_file_name,
    target_number_of_schedules,
    depth_bound,
    frontier_max_size):
    """Initialize a simulation and a virtual world.
        
    Keyword arguments:
    primary_country_actor -- the name of the country that is the primary actor of the simulation
    resource_file_name -- the file path of the resource weights file
    initial_state_file_name -- the file path to the initial world state file
    target_number_of_schedules -- the targeted number of schedules to be generated
    depth_bound -- the maximum depth of traversal for the search
    frontier_max_size -- the maximum size the search frontier can take on

    Returns:
    virtual_world -- an instance of VirtualWorld representing the world being simulated
    """
    # Load the initial world and resource files into memory
    world_state = load_initial_state_file(initial_state_file_name)
    resource_weights = load_resources_file(resource_file_name)

    # Load transform templates
    template_dir = './templates'
    file_list = os.listdir(template_dir)

    template_file_names = []
    
    for f in file_list:
        path_name = template_dir + "/" + f
        template_file_names.append(path_name)

    transform_templates = []

    for template in template_file_names:
        transform_template = parse_transform_template(template)
        transform_templates.append(transform_template)

    # Instantiate a virtual world
    virtual_world = VirtualWorld(initial_state=world_state, 
        resource_weights=resource_weights,
        transform_templates=transform_templates,
        self_country_name=primary_country_actor, 
        target_num_schedules=target_number_of_schedules, 
        depth_bound=depth_bound, 
        frontier_size_limit=frontier_max_size)
    
    return virtual_world


if __name__ == "__main__":
    DEFAULT_COUNTRY_NAME = "Atlantis"
    DEFAULT_RESOURCE_FILE = "./virtual_worlds/project_writeup_example/resources.csv"
    DEFAULT_INITIAL_STATE_FILE = "./virtual_worlds/project_writeup_example/initial_state.csv"

    DEFAULT_OUTPUT_SCHEDULE_FILE = "./simulation_output/output.txt"
    DEFAULT_NUM_SCHEDULES = 1
    DEFAULT_DEPTH_BOUND = 3
    DEFAULT_FRONTIER_SIZE = 5

    print("Calling Country Scheduler")
    country_scheduler(
        your_country_name=DEFAULT_COUNTRY_NAME, 
        resources_filename=DEFAULT_RESOURCE_FILE,
        initial_state_file_name=DEFAULT_INITIAL_STATE_FILE, 
        output_schedule_file_name=DEFAULT_OUTPUT_SCHEDULE_FILE,
        num_output_schedules=DEFAULT_NUM_SCHEDULES, 
        depth_bound=DEFAULT_DEPTH_BOUND, 
        frontier_max_size=DEFAULT_FRONTIER_SIZE)
    
