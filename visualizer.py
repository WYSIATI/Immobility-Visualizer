import argparse
from collections import defaultdict
import cv2
import datetime
import numpy as np

# RGB decimals
RED = (0, 0, 255)
GREEN = (0, 255, 0)

# Kernal size and standard deviation for Gaussian blurring process.
KERNAL_SIZE = (49, 49)
STANDARD_DEVIATION = 0

# Font definitions
NORMAL = 1
BOLD = 2
FONT_SIZE = 0.5

# X coordinate of texts on the frame.
X_COORD = 10

# Threshhold value.
THRESH_VAL = 25

# Maximum value to use with the THRESH_BINARY
# and THRESH_BINARY_INV thresholding types.
MAX_VAL = 255

# The delay of key waiting in cv2
DELAY = 1

# The method and the threshold to compare histograms.
# Method value 0 stands for correlation (CV_COMP_CORREL).
COMP_METHOD = 0
HIST = 0.5

class Target(object):
    """Target monitoring system for limited area."""

    def __init__(self):
        # Construct the argument parser and parse the arguments.
        ap = argparse.ArgumentParser()
        ap.add_argument('-v', '--video', help='path to the video file')
        ap.add_argument('-a', '--min-pixels', type=int,
                        default=5000, help='minimum changed pixel number')
        self._args = vars(ap.parse_args())

        # If the video argument is None, then we are reading from webcam.
        # Otherwise, we are reading from a video file.
        if self._args.get('video', None) is None:
            self._camera = cv2.VideoCapture(0)
        else:
            self._camera = cv2.VideoCapture(self._args['video'])

    def build_hist(self):
        """This function reads a list of image names to build
        histograms for objects. It return a dictionary whose keys
        and values are object names and histograms respectively."""

        # Read the file that contains image names.
        img_names = open('images/list', 'r')

        # Create histograms for objects
        histograms = defaultdict(list)
        for name in img_names:
            temp_name = name.replace('\n', '')
            if len(temp_name) > 1:
                img = cv2.imread('images/%s.png' % temp_name, 0)
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                name_no_digit = filter(lambda c: not c.isdigit(), temp_name)
                histograms[name_no_digit].append(hist)

        return histograms

    def run(self):
        # Get histograms.
        histograms = self.build_hist()

        # Initialize the first frame of the video stream.
        first_frame = None

        # Loop over the frames of the video.
        while True:
            # Grab the current frame and initialize the occupied / unoccupied text.
            grabbed, frame = self._camera.read()
            text = 'Unchanged'

            # If the frame can not be grabbed, then
            # we have reached the end of the video.
            if grabbed is None:
                break

            # Convert the frame to grayscale and blur itself.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            temp_gray = gray
            gray = cv2.GaussianBlur(gray, KERNAL_SIZE, STANDARD_DEVIATION)

            # If the first frame is None, initialize the first frame.
            if first_frame is None:
                first_frame = gray
                continue

            # Compute the absolute difference between
            # the current frame and the first frame.
            frame_delta = cv2.absdiff(first_frame, gray)
            thresh = cv2.threshold(frame_delta, THRESH_VAL, MAX_VAL, cv2.THRESH_BINARY)[1]

            # Dilate the thresholded image to fill in holes,
            # then find contours on the thresholded image.
            thresh = cv2.dilate(thresh, None, iterations = 2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Ignore the contour that has less changed pixels than
                # default or command-line-defined pixels number.
                if cv2.contourArea(contour) < self._args['min_pixels']:
                    continue

                # Compute the bounding box for the contour, draw
                # it on the frame, and update the text.
                x, y, w, h = cv2.boundingRect(contour)
                # Filter out the hand and any other objects that connect
                # to the edge of the frame.
                if x == 1 or y == 1:
                    continue

                # Find the mask and build a histogram for the object.
                mask = np.zeros(frame.shape[:2], np.uint8)
                mask[y:y + h, x:x + w] = 255
                masked_img = cv2.bitwise_and(frame, frame, mask = mask)
                obj_hist = cv2.calcHist([frame], [0], mask, [256], [0, 256])

                # Compare the current object histogram with stored histograms.
                current_min = [None, float('inf')]
                for name in histograms.keys():
                    # Find the minimum distance histogram
                    for hist in histograms[name]:
                        retval = cv2.compareHist(obj_hist, hist, COMP_METHOD)
                        if retval < current_min[1] and retval > 0.1:
                            current_min = name, retval

                if current_min[1] < HIST:
                    # Draw the object boundary box.
                    cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN)
                    # Tag the object.
                    cv2.putText(frame, '%s' % current_min[0], (x - 25, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2,
                                cv2.CV_AA)

                    text = 'Changed'

            # Draw the text and timestamp on the frame.
            cv2.putText(frame, 'Status: {}'.format(text),
                        (X_COORD, 2 * X_COORD), cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SIZE, RED, BOLD)
            cv2.putText(frame,
                        datetime.datetime.now().strftime('%A %d %B %Y '
                                                         '%I:%M:%S%p'),
                        (X_COORD, frame.shape[0] - X_COORD),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE,RED, NORMAL)

            # Show the frame and record if the user presses a key.
            cv2.imshow('Visualizer Feed', frame)

            # Wait for a exit signal that is a letter `q` or `Q` from the user.
            key = cv2.waitKey(DELAY) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break

        # Clean up the camera and close any open windows.
        self._camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    target = Target()
    target.run()
