"""Console Implementation of the Polynomial Regression Model.
"""

from os.path import isfile
from argparse import ArgumentParser
from polynomialRegression import PolyomialRegressionModel
from dataset import Dataset


def parse_arguments() -> tuple[str]:
    """ Parses console arguments for the filepaths of the training and testing 
        dataset of the polynomial solver model.

    Returns:
        train_dataset (string:  Filepath of a Nx2 .csv file with x, y per row
        train_dataset (string): Filepath of a Nx2 .csv file with x, y per row
    """

    parser = ArgumentParser(prog='Polynomial Solver')

    parser.add_argument('--train',  default="data\data_train.csv",
                        type=str, help="File path of the csv document containing the training data.")

    parser.add_argument('--test', default="data\data_test.csv",
                        type=str, help="File path of the csv document containing the testing data.")

    train_csv_filepath = parser.parse_args().train

    assert isfile(train_csv_filepath) and train_csv_filepath.endswith(".csv"), \
        "Training dataset filepath doesn't lead lead to a valid csv document"

    test_csv_filepath = parser.parse_args().test

    assert isfile(test_csv_filepath) and test_csv_filepath.endswith(".csv"), \
        "Testing dataset filepath doesn't lead lead to a valid csv document"

    return train_csv_filepath, test_csv_filepath


def main(train_csv_filepath, test_csv_filepath) -> None:
    """ Loads the training and testing data from the filepaths of .csv files
        with x,y pairs per row, trains a polynomial regression model using the
        training data, and then logs the polynomial degree and coefficients
        with the least loss in reference to the testing data.

    Args:
        train_csv_filepath (str): Filepath of the .csv containing the train x, y
        test_csv_filepath  (str): Filepath of the .csv containing the test x, y
    """

    print("== Dataset ========================================================")

    train_dataset = Dataset(csv_source=train_csv_filepath)
    print(f"Training Dataset Loaded: {len(train_dataset.input)}")

    test_dataset = Dataset(csv_source=test_csv_filepath)
    print(f"Testing Dataset  Loaded: {len(test_dataset.input)}")

    print("== Model Intialization ============================================")
    model = PolyomialRegressionModel()
    # print(model)

    print("== Model Training =================================================")
    # model.train(train_dataset=train_dataset)

    print("== Model Testing ==================================================")
    # model.evaluate(test_dataset=test_dataset)
    # print(model)


if __name__ == '__main__':
    arguments = parse_arguments()
    main(*arguments)
