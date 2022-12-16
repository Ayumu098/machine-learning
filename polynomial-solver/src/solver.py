"""Console Implementation of the Polynomial Regression Model.
"""

from os.path import isfile
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from polynomialRegression import PolynomialRegressionModel as polynomial_model
from dataset import Dataset

def parse_arguments() -> Namespace:
    """Parses console arguments into an argparse Namespace

    returns:
        arguments (Namespace): Contains the console arguments: 
            .test_filepath:  .csv filepath for Nxd input-output train pairs
            .train_filepath: .csv filepath for Nxd input-output test pairs
            .epochs (int): SGD Parameter Optimization steps
            .learning_rate (float): SGD Parameter Optimization step size 
            .verbose (bool): Prints model processes console if True. 
    """

    parser = ArgumentParser(prog='Polynomial Solver',
                            description="Program that trains Polynomial Regression Models from degree zero to degree four with a provided training dataset and determines the optimal degree and polynomial coeeficients for said dataset.")

    # Dataset .csv Filepath: Training
    parser.add_argument('--train_filepath',  default="data\data_train.csv",
                        type=str, help="Filepath of a NxD .csv file with input-output columns representing the training dataset for a polynomial phenomenon. Defaults to data\data_train.csv.")

    # Dataset .csv Filepath: Testing
    parser.add_argument('--test_filepath', default="data\data_test.csv",
                        type=str, help="Filepath of a NxD .csv file with input-output columns representing the unobserved extension of the training dataset. Defaults to data\data_test.csv.")

    # Model Parameter: Stochastic Gradient Descent Steps
    parser.add_argument('--epochs', '--steps', default=1000,
                        type=int, help="Number of iterations for the Stochastic Gradient Descent based model parameter optimization. Defaults to 1000.")

    # Model Parameter: Stochastic Gradient Descent Learning Rate
    parser.add_argument('--learning_rate', '--lr', default=0.001,
                        type=float, help="Scalar of the Gradient in model parameter optimization. Defaults to 0.001.")

    # Program: Display Model Processes in Console
    parser.add_argument('--verbose', '-v', action=BooleanOptionalAction,
                        help="Prints the model processes and status to console if True. Default is False.")

    # Parse console arguments
    arguments = parser.parse_args()

    # Double check validity of csv filepaths
    train_filepath = arguments.train_filepath
    assert isfile(train_filepath) and train_filepath.endswith(".csv"), \
        "Training dataset filepath doesn't lead lead to a valid csv document"
    
    test_filepath = arguments.test_filepath
    assert (isfile(test_filepath) and test_filepath.endswith(".csv")), \
        "Testing dataset filepath doesn't lead lead to a valid csv document"

    return arguments

def main(arguments: Namespace) -> None:
    """ Loads train and test dataset .csv filepaths to Dataset instances,initializes PolynomialRegressionModels with degree range [0, 4],
    trains each model with the training data set with SGD weight optimization,
    and determines the most accurate model using the testing dataset.
    
    Args:
        arguments (Namespace): Contains the console arguments: 
            .test_filepath:  .csv filepath for Nxd input-output train pairs
            .train_filepath: .csv filepath for Nxd input-output test pairs
            .epochs (int): SGD Parameter Optimization steps
            .learning_rate (float): SGD Parameter Optimization step size 
            .verbose (bool): Prints model processes console if True. 
    """

    ## Dataset Loading
    if arguments.verbose:
        print('{0:=<80}'.format('== Dataset Loading '))

    train_dataset = Dataset(csv_source=arguments.train_filepath)
    test_dataset  = Dataset(csv_source=arguments.test_filepath)

    if arguments.verbose:
        print(f"Datasets Loaded: " +
            f"{len(train_dataset.input)} | {len(test_dataset.input)} " +
            "train-test entries")

    ## Model Initialization

    if arguments.verbose:
        print('{0:=<80}'.format('== Polynomial Regression Models '))

    models = [polynomial_model(degree=degree)
              for degree in range(polynomial_model.HIGHEST_DEGREE+1)]

    ## Model Iteration

    for degree, model in enumerate(models):

        # Model Initial Weights
        if arguments.verbose:
            print('{0:=<80}'.format(f'== Degree {degree} Polynomial Model '))
            print(f"Initial Weights:{model}")

        # Model Training
        if arguments.verbose:
            print('{0:=<80}'.format(""))

        model.train(train_dataset=train_dataset,
                    learning_rate=arguments.learning_rate,
                    epochs=arguments.epochs,
                    verbose=arguments.verbose)

        if arguments.verbose:
            print('{0:=<80}'.format(""))

        # Model Final Weights
        model.test(test_dataset=test_dataset)

        # Model Accuracy
        if arguments.verbose:
            print("Accuracy: " +
                 f"{model.accuracy:5.5%} achieved within {arguments.epochs} epochs")
            print("Loss Range: " +
                 f"[{model.minimum_loss:5.5e} - {model.maximum_loss:5.5e}]")

    ## Optimal Model Selection
    optimal_model = min(models, key=lambda model: abs(model.accuracy-1))

    print()
    print('{0:=<80}'.format( "== Optimal Polynomial Model "))
    print('{0:=<80}'.format(f"== Degree {optimal_model.degree} Polynomial"))
    print(optimal_model)
    print(f"Accuracy: {optimal_model.accuracy:5.5%}")
    print("Loss Range: " +
        f"[{optimal_model.minimum_loss:5.5e} - " +
        f"{optimal_model.maximum_loss:5.5e}]")
    
if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
