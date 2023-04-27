# AI Powered Country Trading Simulation
Semester project for CS 5260 - Artificial Intelligence at Vanderbilt

## Set up
Clone this repository by running: **git clone https://github.com/justws95/cs5260-programming-project.git**

Install all requirements by running: **pip install -r requirements.txt**

**NOTE**: _I tested and developed using Python version 3.9.12 and cannot guarantee that it
will work with older versions of python._

## Executing
The code can be executed by navigating to the top-level directory of this repository and running
**python main.py**

**NOTE**: _The program is written in and only compatible with Python 3_

The program will search for n number of schedules where n is supplied in **main.py**. The event loop can be terminated
early by the user by pressing Ctrl + c. If any schedules have been found, they will be written to the output file. If 
none have been found yet, no output files will be created.

Output files are generated and placed in the simulation_output direction, located just below the top level of the
repository. Each run of the program file will create a timestamped sub-directory. The output file, name specified in **main.py**
will be placed inside this lower directory. 


## Changing Program Inputs
I did not have the time to implement a CLI for changing inputs to the program like I wanted to. For
now, program inputs must be changed by altering main.py. At the bottom of the file is the following
section

```python
if __name__ == "__main__":
    DEFAULT_COUNTRY_NAME = "Atlantis"
    DEFAULT_RESOURCE_FILE = "./virtual_worlds/add_food_resource/resources.csv"
    DEFAULT_INITIAL_STATE_FILE = "./virtual_worlds/add_food_resource/initial_state.csv"
    DEFAULT_OUTPUT_SCHEDULE_FILE = "output.txt"
    DEFAULT_NUM_SCHEDULES = 1
    DEFAULT_DEPTH_BOUND = 40
    DEFAULT_FRONTIER_SIZE = 1250

    print("Calling Country Scheduler")
    country_scheduler(
        your_country_name=DEFAULT_COUNTRY_NAME, 
        resources_filename=DEFAULT_RESOURCE_FILE,
        initial_state_file_name=DEFAULT_INITIAL_STATE_FILE, 
        output_schedule_file_name=DEFAULT_OUTPUT_SCHEDULE_FILE,
        num_output_schedules=DEFAULT_NUM_SCHEDULES, 
        depth_bound=DEFAULT_DEPTH_BOUND, 
        frontier_max_size=DEFAULT_FRONTIER_SIZE)
```

This section allows you to change the input files for resource and initial states, the primary country (i.e. 'self') and the output file name


## Notes
**main.py** is the entrypoint of the program and sets up the simulation.

**virtual_world.py** is the top level class that contains a model of the virtual world specified by the inputs and executes the simulation.
Most operations are done or called from this class.

**common/** contains modules re-used throughout the program

**templates/** contains Transformation template files that are read in by the program. All templates in this directory will be read in during runtime.
However, transforms that are not possible because the resources they use/create do not exist in the initial world state definition will be ignored.

**util/** contains utilities for reading the template files, initial world state files, and resource weight files.

**logging_utils/** module that contains SimulationLogger class. Implemented in the singleton design pattern, this utility class is imported throughout
the project to assist with logging.

**virtual_worlds/** contains the initial world states and resource weights for the virtual test worlds I used

**simulation_output/** contains output files generated from program executions

**logs/** contains log files generated from program runs

## Key Classes

**VirtualWorld** Handles simulation responsibilities. Acts upon the other classes to carry out a simulation.

**Transfer** Models a Transfer type operation.

**Transform** Models a Tranform type operation.

**StateNode** Node element of a the search tree representing the world state as it exists at this step in time

**ResourceWeights** Represents the relative weights of each resource

**TransformTemplate** Represents a Transform that can take place.

**SimulationLogger** Application logging utility.

