# Homography

## Description

This is a simple project for the derivation, testing and implementation of a polynomial solver that estimates the degree and coefficients of a polynomial function that closely represents an input-output dataset using [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) or SGD from the [tinygrad](https://github.com/geohot/tinygrad) framework.

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

To run the polynomial solver on a training and testing dataset that are in csv format, use the following command and replace the path arguments accordingly. The format of the csv files are expected to be Nx2 wherein the first column are the input values and the second column is the output value. The first row is ignored. 

```console
python Src\solver.py --data_train=path_train --data_test=path_test
```

If no arguments are provided, the default csv files provided will be used.