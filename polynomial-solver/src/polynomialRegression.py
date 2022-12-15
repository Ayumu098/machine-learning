from tinygrad.tensor import Tensor, Function
from tinygrad.nn.optim import SGD, get_parameters
from dataset import Dataset
from tqdm import trange
from extra.training import train

class PolynomialRegressionModel(Function):
    """ Monovariate Prediction Model that fits a given data using Stochastic
        Gradient Descent (SGD) to a polynomial function with Root Mean Square
        Error as the loss function.
    """

    HIGHEST_DEGREE = 5

    def __init__(self, device: str = "CPU", *tensors: Tensor, degree:int):
        """ Initializes the monomial weights of the polynomial model.
            Assumed to be in increasing order of monomial power starting at 0.

        Args:
            degree (int): Highest monomial exponent in the polynomial.
            device (str, optional): Device to run gradients. Defaults to "CPU".
        """
        super().__init__(device, *tensors)

        # Coefficients of the polynomial components
        self._weights = Tensor.glorot_uniform(1, degree+1)

    @property
    def degree(self):
        return self._weights.shape[1]-1

    def __repr__(self) -> str:
        return str(self._weights.data[0])

    def __call__(self, features: Tensor) -> Tensor:
        # Input Matrix Transformation to Polynomial Form
        polynomial_matrix = Tensor.zeros_like(features) + 1

        for degree in range(1,self.degree+1):
            polynomial_matrix = polynomial_matrix.cat(
                features.pow(degree), dim=-1)

        output = polynomial_matrix.matmul(self._weights.transpose())
        return output

    def forward(self, *args, **kwargs):
        prediction  = self(self._training_features)
        self._loss = prediction.sub(self._training_targets).pow(2).mean().sqrt()
    
    def backward(self, *args, **kwargs):
        self._optimizer.zero_grad()
        self._loss.backward()
        self._optimizer.step()

    def train(self, train_dataset: Dataset, learning_rate:float=0.001,epochs:int=1000, verbose=False):

        self._optimizer = SGD(params=get_parameters(self), lr=learning_rate)
        self._training_features = Tensor(train_dataset.input)
        self._training_targets = Tensor(train_dataset.output)
        
        if verbose:
            progress_bar = trange(epochs)
            for _ in progress_bar:
                self.forward()
                self.backward()
                progress_bar.set_description(
                    f"Training: loss: {self._loss.data[0]:5.5e}")
        else:
            for _ in range(epochs):
                self.forward()
                self.backward()

    def test(self, test_dataset: Dataset) -> float:
        testing_features = Tensor(test_dataset.input)
        testing_targets  = Tensor(test_dataset.output)

        total    = testing_targets.sub(testing_targets.mean()).pow(2).sum()
        residual = testing_targets.sub(self(testing_features)).pow(2).sum()

        self.accuracy =  1 - residual.data[0] / total.data[0]
