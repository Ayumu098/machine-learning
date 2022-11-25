class PolyomialRegressionModel():

	def __init__(self) -> None:
		self.weights = []
		self.bias = []

	def parameters(self):
		return NotImplementedError()

	def __repr__(self) -> str:
		return NotImplementedError()

	def train(self, train_dataset, optimizer = None, epoch=1000):
		return NotImplementedError()

	def evaluate(self, test_dataset):
		return NotImplementedError()