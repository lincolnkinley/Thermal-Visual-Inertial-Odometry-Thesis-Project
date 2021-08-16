# Thermal-Visual-Inertial-Odometry-Thesis-Project
Files in this directory were used to calculate visual odometry. 

drone.py is the primary visual odometry algorithm. It will open files and attempt to run visual odometry on them. You will need Python 3 and GTSAM to run this
g2o_calculator.py is the optimization script. It uses the g2o format files that are created in drone.py. You won't need to run this, it's just a function included in drone.py.
g2o_location.py is used to combine the location of all sensors into a single .csv file for analyzing. It uses the g2o file generated during visual odometry to determine the location of the sensors, and the pickled gps data from gps_to_pickle.py located in ros_tools.

 
