"""
GUI Application with CV2 Library. Allows the user to load an image from a file path, select four points on the image, display a perspective transform of the image to canonical bases, and finally save the image to a file path.
"""

import cv2
import argparse
from os.path import isfile
from homography import to_canonical


def record_point(event, x_value: int, y_value: int, _, marked_points: list):
    """
    Records mouse clicks in the markers list. Not meant to be called outside event triggers.

    Args:
            event: event information (not used)
            x_value (int): horizontal-position of the mouse click
            y_value (int): vertical-position   of the mouse click
            _: flags for the event (not used)
            marked_points: currently recorded marks from mouse clicks
    """

    if event == cv2.EVENT_LBUTTONDOWN:
        marked_points.append((x_value, y_value))


def get_file_locations() -> tuple[str]:
    """
    Obtains the file load location of the image to undistort
    and the file save location of the undistorted file from command-line arguments. If arguments are invalid, continually prompt user for input.

    Returns:
        tuple[str]: the input_path and output_path from cli in a tuple
    """

    parser = argparse.ArgumentParser(prog='Homography GUI Application')

    parser.add_argument('--input_path',  default="Src\source.png",
                        type=str, help="File path with file name and extension of the undistorted image to load. Defaults to \"source.png\" in source directory.")

    parser.add_argument('--output_path', default="Src\\target.png",
                        type=str, help="File path with file name and extension of the undistorted image to save. Defaults to \"target.png\" in source directory.")

    arguments = parser.parse_args()
    input_path, output_path = arguments.input_path, arguments.output_path

    while not isfile(input_path):
        input_path = input("Invalid load filepath. Input image file path: ")

    while not isfile(output_path):
        output_path = input("Invalid load filepath. Input image file path: ")

    return input_path, output_path


def main() -> None:
    """Main function"""

    # Window Settings
    SOURCE_WINDOW = "Source Image"
    TARGET_WINDOW = "Target Image"

    # Point markers for corners of the document in the distorted image
    image_corners = []
    MARKER_COLOR_GREEN = (0, 255, 0)

    input_path, output_path = get_file_locations()
    cv2.namedWindow(SOURCE_WINDOW, cv2.WINDOW_KEEPRATIO)
    source = cv2.imread(input_path)
    cv2.imshow(SOURCE_WINDOW, source)

    # The cv2.imread() of the input_path image and output_path image
    source, target = None, None

    try:
        # Continually run while SOURCE_WINDOW isn't closed
        while cv2.getWindowProperty(SOURCE_WINDOW, 0) >= 0:

            # SOURCE WINDOW Settings
            if len(image_corners) == 0:
                source = cv2.imread(input_path)

            for point in image_corners:
                cv2.circle(img=source, center=point, radius=3,
                           color=MARKER_COLOR_GREEN, thickness=cv2.FILLED)

            cv2.imshow(SOURCE_WINDOW, source)

            # TARGET WINDOW Settings
            if len(image_corners) == 4:
                target = to_canonical(cv2.imread(input_path), image_corners)
                cv2.namedWindow(TARGET_WINDOW, cv2.WINDOW_KEEPRATIO)
                cv2.imshow(TARGET_WINDOW, target)
            elif len(image_corners) == 5:
                image_corners.clear()
                cv2.destroyWindow(TARGET_WINDOW)

            # Mouse callback for recording points of mouse click
            cv2.setMouseCallback(SOURCE_WINDOW, record_point, image_corners)
            cv2.waitKey(1)

    except cv2.error:
        cv2.imwrite(output_path, target)


if __name__ == '__main__':
    main()
