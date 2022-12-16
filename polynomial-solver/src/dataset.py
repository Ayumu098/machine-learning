from numpy import genfromtxt, array
from dataclasses import dataclass, field

@dataclass
class Dataset():
    """Contains the input and output values observed from a phenomenon.
    """
    
    csv_source: str
    input_columns:  tuple[int] = tuple([0])
    output_columns: tuple[int] = tuple([1])
    input:  array = field(init=False)
    output: array = field(init=False)

    def __post_init__(self):
        """Loads the input and output pairs from provided csv filepath
        """

        # Extract all data past the header row
        csv_contents = genfromtxt(self.csv_source, delimiter=',')[1:]

        # Set input and output pairs as (N, d) shaped numpy matrices
        self.input  = csv_contents[:, self.input_columns ].reshape(-1, 1)
        self.output = csv_contents[:, self.output_columns].reshape(-1, 1)

        # Format input and output as (N, 1) shaped numpy arrays
        self.input  = array(self.input).reshape(-1, 1)
        self.output = array(self.input).reshape(-1, 1)