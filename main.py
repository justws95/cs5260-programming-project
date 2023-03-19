"""Main entry point for the program."""


import os
from util import load_initial_state_file, load_resources_file, parse_transform_template



def country_scheduler(your_country_name, resources_filename,
initial_state_filename, output_schedule_filename,
num_output_schedules, depth_bound,
frontier_max_size):
    """Top level method for running a simulation."""
    init_simulation


def init_simulation():
    """Initialize a simulation."""
    input_world_file = './virtual_worlds/project_writeup_example/initial_state.csv'
    input_resources_file = './virtual_worlds/project_writeup_example/resources.csv'

    world_state = load_initial_state_file(input_world_file)
    resource_weights = load_resources_file(input_resources_file)

    # load transform templates
    template_dir = './templates'
    file_list = os.listdir(template_dir)

    template_list = []

    for f in file_list:
        path_name = template_dir + "/" + f
        template_list.append(path_name)


    for template in template_list:
        transform_template = parse_transform_template(template)


if __name__ == "__main__":
    DEFAULT_COUNTRY_NAME = "Atlantis"
    DEFAULT_RESOURCE_FILE = "./virtual_worlds/project_writeup_example/resources.csv"
    DEFAULT_INITIAL_STATE_FILE = "./virtual_worlds/project_writeup_example/initial_state.csv"

    DEFAULT_OUTPUT_SCHEDULE_FILE = "./simulation_output/output.txt"
    DEFAULT_NUM_SCHEDULES = 1
    DEFAULT_DEPTH_BOUND = 3
    DEFAULT_FRONTIER_SIZE = 5

    country_scheduler(
        your_country_name=DEFAULT_COUNTRY_NAME, 
        resources_filename=DEFAULT_RESOURCE_FILE,
        initial_state_filename=DEFAULT_INITIAL_STATE_FILE, 
        output_schedule_filename=DEFAULT_OUTPUT_SCHEDULE_FILE,
        num_output_schedules=DEFAULT_NUM_SCHEDULES, 
        depth_bound=DEFAULT_DEPTH_BOUND, 
        frontier_max_size=DEFAULT_FRONTIER_SIZE)
    
