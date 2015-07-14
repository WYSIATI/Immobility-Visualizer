import argparse
import cv2
import datetime
import numpy as np
from matplotlib import pyplot as plt

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
COMP_CHISQR = 1
HIST = 50000

class Target(object):
	"""Target monitoring system for limited area."""

	def __init__(self):
		# Construct the argument parser and parse the arguments.
		ap = argparse.ArgumentParser()
		ap.add_argument('-v', '--video', help='path to the video file')
		ap.add_argument('-a', '--min-pixels', type=int,
						default=300, help='minimum changed pixel number')
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
		histograms = dict()
		for name in img_names:
			if len(name) > 1:
				img = cv2.imread('images/%s' % name[:-1], 0)
				hist = cv2.calcHist([img], [0], None, [256], [0, 256])
				histograms[name[7:-1]] = hist

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

			# Iterate through the contours.
			for contour in contours:
				# Ignore the contour that has less changed pixels than
				# default or command-line-defined pixels number.
				if cv2.contourArea(contour) < self._args['min_pixels']:
				    continue

				# Compute the bounding box for the contour, draw
				# it on the frame, and update the text.
				x, y, w, h = cv2.boundingRect(contour)
				# Filter out the hand.
				if x == 1 or y == 1 or w == 1 or h == 1:
					continue

				# Find the mask and build a histogram for the object.
				mask = np.zeros(temp_gray.shape[:2], np.uint8)
				mask[y:y + h, x:x + w] = 255
				masked_img = cv2.bitwise_and(temp_gray, temp_gray, mask = mask)
				# plt.imshow(masked_img)
				# plt.show()
				obj_hist = cv2.calcHist([temp_gray], [0], masked_img, [256], [0, 256])

				# Compare the current object histogram with stored histograms.
				for name in histograms.keys():
					hist = histograms[name]
					print cv2.compareHist(obj_hist, hist, COMP_CHISQR)
					if cv2.compareHist(obj_hist, hist, COMP_CHISQR) < HIST:
						cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN)

						# Tag the object.
						cv2.putText(frame, '%s' % name, (x - 25, y - 10),
									cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2,
									cv2.CV_AA)

						text = 'Changed'
						break

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
			cv2.imshow('Security Feed', frame)

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