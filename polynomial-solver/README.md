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

The tinygrad framework is included in the Src path with minimal dependencies.

## Usage

### Training and Testing Dataset

To run the polynomial solver on a training and testing dataset that are in csv format, use the following command and replace the path arguments accordingly. The format of the csv files are expected to be Nx2 wherein the first column are the input values and the second column is the output value. The first row is ignored. 

```console
python src\solver.py --train=path_train --test=path_test
```

If no arguments are provided, the default csv files provided, `data\data_test.csv` and `data\data_train.csv`, will be used.

### Model Learning Parameters

The iterations or epochs taken by the model training process and the learning rate can also be optionally configured as shown.
```console
python src\solver.py --epochs=int_value --learning_rate=float_value
```

### Logging

To see the processes done by each model initialized, add a `--verbose` command.

```console
python src\solver.py --verbose
```

## Metrics

### Loss Function

The [Root Mean Square Loss](https://en.wikipedia.org/wiki/Coefficient_of_determination) is used for the SGD-based model training given the relatively straightforward implementation of its algorithm in the tinygrad framework.

### Model Accuracy

The [R2 Loss](https://en.wikipedia.org/wiki/Coefficient_of_determination) is used to determine model accuracy as it represents the accuracy more intuitively in terms of percentage (0 to 100% accuracy).