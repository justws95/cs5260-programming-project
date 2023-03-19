"""Model representing a Transformation Template."""


import re



class TransformTemplate:
    """Model a transformation template for a given transform."""

    def __init__(self, name, transform_file_contents):
        """Initialize a TransformTemplate instance."""
        self.transform_name = name
        self.inputs = {}
        self.outputs = {}

        self._read_in_transform_file(transform_file_contents)

        # Apply helpers to only look at keys for inputs/outputs
        self._input_list = []
        self._output_list = []

        for key, _ in self.inputs.items():
            self._input_list.append(key)

        for key, _ in self.outputs.items():
            self._output_list.append(key)

        return
    
    def __repr__(self):
        """Define the printed representation of this transform template."""
        REPR = ""

        REPR += f"{self.transform_name}\n"
        REPR += "-"*20 + "\n"
        REPR += "INPUTS\n"
        for key, val in self.inputs.items():
            REPR += f"  {key}: {val}\n"
        REPR += "\n"
        REPR += "OUTPUTS\n"
        for key, val in self.outputs.items():
            REPR += f"  {key}: {val}\n"
        REPR += "\n\n"
        
        return REPR
    

    def _extract_resource_mapping_from_text(self, text_block):
        """Use regular expressions to parse text block for resource names and quantities.
        
        Keyword arguments:
        text_block -- the block of text to parse

        Returns:
        resource_mapping -- a Python Dictionary mapping a resource to its quantity
        """
        resource_mapping = {}

        regex = r"\(([A-Za-z]+) (\d)\)"
        matches = re.finditer(regex, text_block, re.MULTILINE)

        for match in matches:
            resource_name, resource_quantity = match.groups()
            resource_mapping[resource_name] = resource_quantity

        return resource_mapping
    

    def _read_in_transform_file(self, transform_file_contents):
        """Read in a transform file's contents and map internally.
        
        Keyword arguments:
        transform_file_contents -- the text contents of the template file
        """
        # Find the input and output text blocks
        inputs_start = transform_file_contents.index("INPUTS")
        outputs_start = transform_file_contents.index("OUTPUTS")

        inputs_string = transform_file_contents[inputs_start:outputs_start]
        outputs_string = transform_file_contents[outputs_start:]

        # Extract the resource quantities from the text
        inputs_dict = self._extract_resource_mapping_from_text(inputs_string)
        outputs_dict = self._extract_resource_mapping_from_text(outputs_string)

        self.inputs = inputs_dict
        self.outputs = outputs_dict
        
        return
    

    def get_inputs_list(self):
        """Get the list of inputs for this transform.

        Returns:
        input_list -- a list of all the transform inputs
        """
        return self._input_list
        

    def get_outputs_list(self):
        """Get the list of inputs for this transform.

        Returns:
        output_list -- a list of all the transform outputs
        """
        return self.self._output_list
        
