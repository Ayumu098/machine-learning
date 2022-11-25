from numpy import genfromtxt
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Dataset():
    """Contains the input and output values observed from a function.
    """

    input:  list = Optional[List[float]]
    output: list = Optional[List[float]]

    def __init__(self, csv_source: str):
        """Loads the input and output pairs from a .csv document filepath.

        Args:
            csv_source (str): Filepath to a Nx2 .csv file containing the 
            x, y (input-output) pairs for a function.
        """

        self.load(csv_source=csv_source)

    def load(self, csv_source: str):
        """Loads the input and output pairs from a .csv document filepath.

        Args:
            csv_source (str): Filepath to a Nx2 .csv file containing the 
            x, y (input-output) pairs for a phenomenon to be modelled.
        """

        # Reads the rows past the header of the csv file as two tuples
        input_output_pair = zip(*genfromtxt(csv_source, delimiter=',')[1:])

        # Load the input and outputs as type list for convenience
        self.input, self.output = map(list, input_output_pair)

        assert len(self.input) == len(self.output),\
            f"Input count in {csv_source} must match Output count"