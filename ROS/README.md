# Thermal-Visual-Inertial-Odometry-Thesis-Project
This is the ROS node used to process data collected by the cameras. 
Each of the cameras have an *_processor.py file that will process their images.
It also has two early methods that were developed for detecting motion. 
optical_flow.py attempts to determine motion though dense optical flow.
drone_translation.py attempts to determine motion through sparse optical flow. 
Neither of these files were used for data processing the final data presented in the thesis, but they do represent part of the journey though the thesis. 
These are set up to work with the lepton camera.
A Launch file is provided that will launch all of these files in ROS.

In order to run these, you will need python 2.7, ROS, OpenCV 4.5.3 (older versions might work, but no promises), CV Bridge, numpy, and scipy.
The only step required is dropping the src file into a catkin workspace and running "catkin build".
