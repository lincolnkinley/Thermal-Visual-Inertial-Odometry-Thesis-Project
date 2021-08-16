# Thermal-Visual-Inertial-Odometry-Thesis-Project
The programs in this directory represent tools that were developed to process rosbags.
Data was collected into rosbags, which allowed offline processing. Steps had to be taken to separate ROS from GTSAM, as ROS uses python 2 and GTSAM uses python 3.

gps_to_pickle.py will read gps data from a rosbag and save it as a python pickle. This data can later be opened later by python 3.
LeptonBosonBlackfly2Video.py will open a rosbag and create a video with all three cameras that synchronizes their timing.
post_process_rosbag.py will run contrasting on all images. The ProcessBlackflyMessage() function was used when the blackfly was collecting images at nighttime, but not at daytime.
project_with_bah.py will open a rosbag, undistort the images, project the images using a static transform and the attitude of the IMU, and then save them as a .png file. This was the last step for ROS processing, and the images would be opened in python 3 for visual odometry.
repair_rosbag_timing.py will open a rosbag and set the publishing time of each message to the time in the header. This tool was used to fix a rosbag that was processed but used computer time and not ros time as the header of all messages. This feature was added to post_process_rosbag.py later.
