"""Console Implementation of the Polynomial Regression Model.
"""

from os.path import isfile
from argparse import ArgumentParser, BooleanOptionalAction
from polynomialRegression import PolynomialRegressionModel as polynomial_model
from dataset import Dataset

def parse_arguments() -> tuple[str]:
    """ Parses console arguments for the filepaths of the training and testing
        dataset of the polynomial solver model.

    Returns:
        train_dataset (string:  Filepath of a Nx2 .csv file with x, y per row
        train_dataset (string): Filepath of a Nx2 .csv file with x, y per row

    """

    parser = ArgumentParser(prog='Polynomial Solver')

    ## Command Line Interface Arguments

    parser.add_argument('--train',  default="data\data_train.csv",
                        type=str, help="File path of the csv document containing the training data.")

    parser.add_argument('--test', default="data\data_test.csv",
                        type=str, help="File path of the csv document containing the testing data.")

    parser.add_argument('--epochs', default=1000,
                        type=int, help="Number of iterations for the Stochastic Gradient Descent process.")

    parser.add_argument('--learning_rate', default=0.001,
                        type=float, help="Learning rate or scalar of the gradient applied to the weights during the Stochastic Gradient Descent process.")

    parser.add_argument('--verbose', action=BooleanOptionalAction, 
                        help="Log the model loading and processes to console if enabled.")

    ## Dataset Filepaths

    train_csv_filepath = parser.parse_args().train

    assert isfile(train_csv_filepath) and train_csv_filepath.endswith(".csv"), \
        "Training dataset filepath doesn't lead lead to a valid csv document"

    test_csv_filepath = parser.parse_args().test

    assert isfile(test_csv_filepath) and test_csv_filepath.endswith(".csv"), \
        "Testing dataset filepath doesn't lead lead to a valid csv document"

    ## Model Components
    epochs  = parser.parse_args().epochs
    learning_rate = parser.parse_args().learning_rate
    verbose = parser.parse_args().verbose

    return train_csv_filepath, test_csv_filepath, epochs, learning_rate, verbose


def main(train_csv_filepath, test_csv_filepath, epochs, learning_rate, verbose):
    """ Loads the training and testing data from the filepaths of .csv files
        with x,y pairs per row, trains a polynomial regression model using the
        training data for given iterations and learning rate, and then  
        determines the polynomial degree and coefficients with the least loss in reference to the testing data.

    Args:
        train_csv_filepath (str): Filepath of the .csv containing the train x, y
        test_csv_filepath  (str): Filepath of the .csv containing the test  x, y
        epochs (int): Number of iterations in model training via SGD.
        learning_rate (float): Determines the step size during model training.
        verbose (bool): Log the model loading and processes to console if True. 
    """

    ## Dataset Loading

    if verbose:
        print('{0:=<50}'.format('== Dataset Testing '))

        train_dataset = Dataset(csv_source=train_csv_filepath)
        print(f"Training Dataset Loaded: {len(train_dataset.input)}\t entries")

        test_dataset = Dataset(csv_source=test_csv_filepath)
        print(f"Testing  Dataset Loaded: {len(test_dataset.input)}\t entries")
    else:
        train_dataset = Dataset(csv_source=train_csv_filepath)
        test_dataset = Dataset(csv_source=test_csv_filepath)

    models = [ polynomial_model(degree=degree) 
        for degree in range(polynomial_model.HIGHEST_DEGREE+1) ]

    if verbose:
        print('{0:=<50}'.format('== Polynomial Regression Models '))

    for degree, model in enumerate(models):

        ## Model Initial Weights
        if verbose:
            print(f"Degree {degree} Polynomial Regression Model")
            print(f"Weights: {model}")

        ## Model Training
        model.train(train_dataset=train_dataset,
            learning_rate=learning_rate, 
            epochs=epochs,
            verbose=verbose)

        if verbose: 
            print(f"Weights: {model}")
    
        ## Model Testing
        model.test(test_dataset=test_dataset)

        if verbose:
            print(f"Accuracy: {model.accuracy:5.5%}")

    optimal_model = min(models, key=lambda model: abs(model.accuracy-1))
    print('{0:=<50}'.format('== Optimal Polynomial Regression Model '))

    print("== Optimal Polynomial Regression Model ")
    print(f"Degree {optimal_model.degree} Weights: {optimal_model}")
    print(f"Accuracy: {optimal_model.accuracy:5.5%}")

if __name__ == '__main__':
    arguments = parse_arguments()
    main(*arguments)
