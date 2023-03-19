"""Main entry point for the program."""

from util import load_initial_state_file, load_resources_file

def country_scheduler(your_country_name, resources_filename,
initial_state_filename, output_schedule_filename,
num_output_schedules, depth_bound,
frontier_max_size):
    pass

def init_simulation():
    """Initialize a simulation."""
    input_world_file = './virtual_worlds/project_writeup_example/initial_state.csv'
    input_resources_file = './virtual_worlds/project_writeup_example/resources.csv'

    world_state = load_initial_state_file(input_world_file)
    resource_weights = load_resources_file(input_resources_file)


if __name__ == "__main__":
    print("Starting the program.")
    init_simulation()
    #country_scheduler()
