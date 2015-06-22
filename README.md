# Immobility-Visualizer
This is a visualizer to recognize objects and track their motion within a relatively small area.

Requirements:

	Python: 2.7.x
	OpenCV: 2.4.x

Usage: Python detector.py [-h] [-v VIDEO] [-a MIN_PIXELS]

Optional arguments:

  	-h, --help
  		Show this help message and exit.
  	-v --video <VIDEO>
  		Path to the video file that is for testing with a recorded video.
  	-a --min-pixels <MIN_PIXELS>
  		Minimum changed pixel number that is for determining contours.
  		
To quit the Python window, press `q` or `Q` to quit the program.
