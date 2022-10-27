"""Module for GUI Application with Tkinter and CV2 Library.
Allows the user to load an image, select four points on the image,
displays a perspective transform of the image to canonical bases,
and finally allows the user to save the image.
"""

import tkinter
import tkinter.filedialog

import cv2
from numpy import ndarray
from homography import to_canonical

MAIN_WINDOW_NAME = "Source Image"
OUTPUT_WINDOW_NAME = "Target Image"

def record_point(event, x_value: int, y_value: int, _, marked_points: list):
    """Records mouse clicks in the markers list. Not meant to be called outside event triggers.

    Args:
            event: event information (not used)
            x_value (int): horizontal-position of the mouse click
            y_value (int): vertical-position   of the mouse click
            _: flags for the event (not used)
            markers: currently recorded marks from mouse clicks
    """

    if event == cv2.EVENT_LBUTTONDOWN:
        marked_points.append((x_value, y_value))

        if len(marked_points) == 5:
            marked_points.clear()


def _load_file_name():
    """Opens a tkinter file dialog window to get the image to process.

    Returns:
        str: filepath of the image to process. Window closed if equal to ''
    """

    filepath = None

    # Load image using tkinter filedialog() window
    loading_window = tkinter.Tk()

    while filepath is None:
        try:
            filepath = tkinter.filedialog.askopenfilename()
            cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(MAIN_WINDOW_NAME, cv2.imread(filepath))
        except cv2.error:

            # Empty filepath implies quiting the file explorer window
            if filepath == '':
                break

            filepath = None
            print("Invalid image file. File dialog will restart")

    loading_window.quit()
    return filepath


def _save_file(image: ndarray):

    # Save image using tkinter filedialog() window
    saving_window = tkinter.Tk()
    filepath = None

    while filepath is None:
        try:
            filepath = tkinter.filedialog.asksaveasfilename()
            cv2.imwrite(filepath, image)
        except cv2.error:

            # Empty filepath implies quiting the file explorer window
            if filepath == '':
                break

            filepath = None
            print("Invalid file name. File dialog will restart")

    saving_window.quit()

def main() -> None:
    """Main function"""

    target = None
    marked_points = []
    filepath = _load_file_name()

    try:
        while cv2.getWindowProperty(MAIN_WINDOW_NAME, 0) >= 0:

           # Reload image for a clean copy every reset
            if len(marked_points) == 0:
                source = cv2.imread(filepath)

            # Render every marked points as small green circles
            for point in marked_points:
                cv2.circle(source, point, 3, (0, 255, 0), cv2.FILLED)

            # On completing four corners, display a undistorted (canonical basis) source image
            if len(marked_points) == 4:
                target = to_canonical(source, marked_points)
                cv2.imshow(OUTPUT_WINDOW_NAME, target)

            cv2.imshow(MAIN_WINDOW_NAME, source)

            # Mouse callback for recording points of mouse click
            cv2.setMouseCallback(MAIN_WINDOW_NAME, record_point, marked_points)
            cv2.waitKey(1)

    # Implies closing of the cv2 source image window
    except cv2.error:
        if target is not None:
            _save_file(target)


if __name__ == '__main__':
    main()
