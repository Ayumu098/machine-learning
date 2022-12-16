# Polynomial Regression Model

## Description

This is a simple project for the implementation of a [Polynomial Regression](https://en.wikipedia.org/wiki/Polynomial_regression) model that estimates the optimal degree and coefficients of an assumed polynomial function that closely fits an input-output dataset using [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) or SGD from the [tinygrad](https://github.com/geohot/tinygrad) framework.

## Purpose

This is made as part of the second requirement, "Polynomial Solver using SGD in TinyGrad" in COE197ML Foundations of Machine Learning (AY 2022-2023) by Professor Rowel Atienza.

## Prerequisites

Clone the GitHub repository using git

```console
git clone https://github.com/Ayumu098/machine-learning.git
cd machine-learning
cd polynomial-solver
```

Install the python package dependencies in requirements.txt

`pip install -r requirements.txt`

The tinygrad framework is included in the crc path with minimal dependencies.

## Console Interface

A main console program `solver.py` is provided to facilitate the finding of the optimal polynomial with degree between 1 to 4 for a provided training and testing dataset. For an overview on the console commands, type the following in the console.

```console
python src\solver.py --help
```

### Training Dataset

To provide the filepath of a .csv file for training the polynomial regression model, enter the following command and change the `FILEPATH` accordingly. If the argument isn't used, the default is `data\data_train.csv`

```console
python src\solver.py --train_filepath=FILEPATH
```

Note that the expected .csv file is (N, 2) with the first row ignored (with the assumption that it's a label header). The first column is for the inputs and their corresponding outputs are in the second column.

### Testing Dataset

To provide the filepath of a .csv file for testing the polynomial regression model, enter the following command and change the `FILEPATH` accordingly. If the argument isn't used, the default is `data\data_test.csv`

```console
python src\solver.py --test_filepath=FILEPATH
```

Similar to the training dataset .csv file,  the expected .csv file is (N, 2) with the first row ignored (with the assumption that it's a label header). The first column is for the inputs and their corresponding outputs are in the second column.


### Model Learning Parameters

The number of iterations or epochs taken during the model training or parameter optimization process can be set using the following command. Make sure to change `EPOCH` accordingly. If the argument isn't used, the default is 1000.

```console
python src\solver.py --epochs=EPOCH --learning_rate=float_value
```

Similarly, the learning rate can also be set. Make sure to change `LR`. If the argument isn't used, the default is 0.001.

```console
python src\solver.py --learning_rate=LR
```

### Console Logging

To see the changing model weights and results during the optimization process, add `--verbose` or `--v` to the existing command.

```console
python src\solver.py --verbose
```

## Polynomial Regression Model Class

Alternatively, the `PolynomialRegressionModel` class can be accessed on its own with the `Dataset` class being the only dependency. The model consists of hidden weights and has the default functionalities of models (i.e., forward propagation via `forward()`, backward propagation via `backward()`, prediction or evaluation via direct call, and additional `train` and `test` methos for convenience).





## Metrics

### Loss Function

The [Root Mean Square Loss](https://en.wikipedia.org/wiki/Coefficient_of_determination) is used for the SGD-based model training given the relatively straightforward implementation of its algorithm in the tinygrad framework.

### Model Accuracy

The [R2 Loss](https://en.wikipedia.org/wiki/Coefficient_of_determination) is used to determine model accuracy as it represents the accuracy more intuitively in terms of percentage (0 to 100% accuracy).