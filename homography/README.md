# Homography

## Description

This is a simple project for the derivation, testing and implementation of a homography between two camera perspectives with exactly four correspondence or points for the purpose of undistorting a scanned document or subject.

## Purpose

This is made as part of the first requirement, "Removing Projective Distortion on Images", in COE197ML [Foundations of Machine Learning](https://github.com/roatienza/ml) (AY 2022-2023).

## Prerequisites

Clone the GitHub repository using git and enter the project folder
```
git clone https://github.com/Ayumu098/machine-learning.git
cd machine learning
cd homography
```

Install the python package dependencies in `requirements.txt`
```
pip install -r requirements.txt
```

## GUI Application
Run the python file `homography_gui.py` with the arguments `--input_path` and `--output_path`. Make sure to not include whitespaces. The default input path is "Src/source.png" and the default output path is "Src/output.png"
```
python Src/homography_gui.py --input_path=load_location_distorted_image.png --output_path=save_location_undistorted_image.png
```

## Jupyter Notebook
The derivation of the homography matrix up to a GUI implementation using python-cv2 can be seen in the `homography.ipynb` file. The file also generates a simple distorted image for testing. This notebook can perform the undistortion of a scanned documents itself.

## Library Use
To only obtain the homography_matrix or use a direct noncanonical to canonical perspective transformation function, `to_canonical`, use the `homography.py` file.