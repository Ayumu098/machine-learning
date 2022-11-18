"""Contains the homography_matrix and to_canonical main functions
"""

from numpy import linalg, ndarray, array
import cv2


def arrange_upper_left_clockwise(points: list[tuple[int]]):
    """Arranges list of points (x, y) in the following arrangement:
    upper left -> upper right -> lower right -> lower left. Assumes the (x, y)
	coordinates follow same system as cv2 and PIL.

    Args:
        points (list[tuple[int]]): List of tuples (x, y)
    """

    # Lowest  distance from origin is upper-leftmost  point
    # Highest distance from origin is lower-rightmost point
    distance_sorted_points = sorted(points, key=sum)
    upper_left, lower_right = distance_sorted_points[::3]

    # If remaining points, upper right is point with highest value of x
    lower_left, upper_right = [point for point in sorted(points)
                               if point not in {upper_left, lower_right}]

    # Clockwise arrangement, starting at upper left point
    return [upper_left, upper_right, lower_right, lower_left]


def to_canonical(source: ndarray, source_points: list[tuple[int]]) -> ndarray:
    """Function to convert an image in non canonical basis to canonical basis
	via projective linear transformation

    Args:
            source (ndarray): image loaded using cv2.imread()
            source_points (list[tuple[int]]): list of 4 tuples (x,y)
			corresponding to points in the source image

    Returns:
            ndarray: image in rectangular form or in canonical basis
    """

    # Set source points to clockwise arrangement, starting upper left
    source_points = arrange_upper_left_clockwise(source_points)

    upper_left, upper_right, lower_right, lower_left = source_points
    left = (upper_left[0] + lower_left[0]) // 2
    right = (upper_right[0] + lower_right[0]) // 2

    top = (upper_left[1] + upper_right[1]) // 2
    bottom = (lower_left[1] + lower_right[1]) // 2

    # Set target points to clockwise arrangement, starting upper left
    target_points = [(left, top), (right, top),
                     (right, bottom), (left, bottom)]

    # Apply the homography matrix for undistortion and crop to size
    undistort = homography_matrix(source_points, target_points)
    target = cv2.warpPerspective(source.copy(), undistort, source.shape[1::-1])

    return target[top:bottom, left:right]


def homography_matrix(source_points: list[float], target_points: list[float]):
    """	Returns the projective transformation H for X to X' projection

    Args:
    source_points (list[float]): Points ( x , y ) of X
        target_points (list[float]): Points ( x', y') of X'
    """

    # Assertion checks for valid parameters
    assert len(source_points) == len(target_points), "Non-matching points"
    assert len(source_points) <= 4, "Need more than 3 points"

    # Matrix A for the homogeneos problem: Ah = 0
    intermediate = []

    for (source_x, source_y), (target_x, target_y) in zip(
            source_points, target_points):

        # Odd  rows of intermediary matrix
        intermediate.extend((
            source_x, source_y, 1, 0, 0, 0,
            -source_x*target_x, -source_y*target_x, -target_x
        ))

        # Even rows of intermediary matrix
        intermediate.extend((
            0, 0, 0, source_x, source_y, 1,
            -source_x*target_y, -source_y*target_y, -target_y
        ))

    # Use right singular column with lowest singular value to minimize AH = 0
    intermediate = array(intermediate).reshape((2 * len(source_points), -1))
    _, _, v_singular = linalg.svd(intermediate)
    homography = v_singular[-1].reshape((3, 3))

    return homography
